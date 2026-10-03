# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Declared-point E_sigma first-hit; not the GRAZING V=0 band or G1."""

from __future__ import annotations

from fractions import Fraction

from omnibias.dynamics.e_out_section import e_sigma
from omnibias.dynamics.e_sigma_hit import (
    KILL_L_PACK,
    identity_verdicts,
    report,
    residual_e_sigma_in_group,
    residual_e_sigma_in_lead,
    residual_restart_h,
    residual_restart_v,
)
from omnibias.dynamics.hilbert16_ledger import default_h16_ledger, derived_parent_flags


def test_e_sigma_hit_identities_are_exact() -> None:
    v_coord, height = Fraction(3, 4), Fraction(1, 4)
    rho, nu, c, sigma = Fraction(1, 4), Fraction(1, 16), Fraction(2), Fraction(-1)
    assert residual_restart_v(v_coord) == 0
    assert residual_restart_h(height) == 0
    assert residual_e_sigma_in_lead(v_coord, height, sigma, rho, c) == 0
    assert residual_e_sigma_in_group(v_coord, height, sigma, rho, nu, c) == 0
    assert e_sigma(v_coord, height, sigma, rho, Fraction(0), c) == v_coord - 1 + sigma * rho * height
    assert all(status == "PROVED" for status in identity_verdicts().values())
    assert KILL_L_PACK == (Fraction(9, 25), Fraction(1, 16), Fraction(0))


def test_e_sigma_first_hit_does_not_pass_g1() -> None:
    payload = report().to_payload()
    assert payload["schema"] == "hilbert16-e-sigma-hit-v1"
    assert payload["e_sigma_first_hit"] is True
    assert payload["outgoing_first_hit"] is False
    assert set(payload["pack"]) == {"9/25", "1/16", "0"}
    assert all(row["status"] == "certified" and row["replayed"] for row in payload["pack"].values())
    assert payload["short"]["status"] != "certified"
    assert payload["from_grazing_start"]["status"] != "certified"
    honesty = payload["honesty"]
    assert honesty["e_sigma_first_hit"] is True
    assert honesty["outgoing_first_hit"] is False
    assert honesty["g1_passed"] is False
    assert honesty["full_hilbert16_solved"] is False
    reasons = honesty["_honesty_reasons"]
    assert reasons["g1_passed"]["reason"] == "gate_open"
    assert "e_sigma_first_hit" not in reasons


def test_local_e_sigma_hit_never_earns_a_parent_flag() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_e_sigma_hit")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["full_hilbert16_solved"] is False
    assert flags["hilbert16_part_b_quadratic_solved"] is False
