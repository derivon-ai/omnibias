# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Orbit-aligned E_sigma from the V=1/4 wall box; not Lohner-from-V=0 or G1."""

from __future__ import annotations

from fractions import Fraction

from omnibias.dynamics.e_out_section import e_sigma
from omnibias.dynamics.e_sigma_wall import (
    KILL_L_PACK,
    identity_verdicts,
    report,
    residual_align_h,
    residual_align_v,
    residual_e_align_nu0,
    residual_e_align_start,
)
from omnibias.dynamics.hilbert16_ledger import default_h16_ledger, derived_parent_flags


def test_e_sigma_wall_identities_are_exact() -> None:
    v_coord, height = Fraction(1, 4), Fraction(1, 40)
    rho, nu, c, sigma = Fraction(1, 4), Fraction(1, 16), Fraction(2), Fraction(-1)
    assert residual_align_v(v_coord) == 0
    assert residual_align_h(height) == 0
    assert residual_e_align_nu0(v_coord, height, sigma, rho) == 0
    assert residual_e_align_start(v_coord, height, sigma, rho, nu, c) == 0
    assert e_sigma(v_coord, height, sigma, rho, Fraction(0), c) == Fraction(-121, 160)
    assert e_sigma(v_coord, height, sigma, rho, nu, c) == Fraction(-619519, 819200)
    assert all(status == "PROVED" for status in identity_verdicts().values())
    assert KILL_L_PACK == (Fraction(9, 25), Fraction(1, 16), Fraction(0))


def test_e_sigma_wall_hit_does_not_pass_g1() -> None:
    payload = report().to_payload()
    assert payload["schema"] == "hilbert16-e-sigma-wall-v1"
    assert payload["e_sigma_wall_hit"] is True
    assert payload["outgoing_first_hit"] is False
    assert set(payload["pack"]) == {"9/25", "1/16", "0"}
    assert all(row["status"] == "certified" and row["replayed"] for row in payload["pack"].values())
    assert all(row["contains_align"] for row in payload["walls"].values())
    assert payload["short"]["status"] != "certified"
    assert payload["from_grazing_start"]["status"] != "certified"
    honesty = payload["honesty"]
    assert honesty["e_sigma_wall_hit"] is True
    assert honesty["outgoing_first_hit"] is False
    assert honesty["g1_passed"] is False
    assert honesty["full_hilbert16_solved"] is False
    reasons = honesty["_honesty_reasons"]
    assert reasons["g1_passed"]["reason"] == "gate_open"
    assert "e_sigma_wall_hit" not in reasons


def test_local_e_sigma_wall_never_earns_a_parent_flag() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_e_sigma_wall")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["full_hilbert16_solved"] is False
    assert flags["hilbert16_part_b_quadratic_solved"] is False
