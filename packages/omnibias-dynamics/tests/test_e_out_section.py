# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Matching-chart E_out first-hit; not GRAZING E_sigma or G1."""

from __future__ import annotations

from fractions import Fraction

from omnibias.dynamics.e_out_section import (
    KILL_L_PACK,
    certify_e_out,
    certify_grazing_excluded,
    e_out,
    e_sigma,
    identity_verdicts,
    report,
    residual_e_out_group,
    residual_e_out_lead,
    residual_e_sigma_nu0,
    residual_section_embed,
)
from omnibias.dynamics.hilbert16_ledger import default_h16_ledger, derived_parent_flags


def test_e_out_section_identities_are_exact() -> None:
    v_coord = -Fraction(1, 16)
    height = Fraction(4) * Fraction(1, 16) ** 3
    rho, nu, c, sigma = Fraction(1, 4), Fraction(1, 16), Fraction(2), Fraction(1)
    assert residual_e_sigma_nu0(v_coord, height, sigma, rho, c) == 0
    assert residual_e_out_lead(v_coord, rho, c) == 0
    assert residual_section_embed(Fraction(1, 16), rho) == 0
    assert residual_e_out_group(v_coord, height, rho, nu, c) == 0
    assert e_sigma(v_coord, height, sigma, rho, Fraction(0), c) == v_coord - 1 + sigma * rho * height
    assert e_out(v_coord, Fraction(0), rho, Fraction(0), c) == v_coord + rho
    assert all(status == "PROVED" for status in identity_verdicts().values())


def test_e_out_first_hit_does_not_pass_g1() -> None:
    payload = report().to_payload()
    assert payload["schema"] == "hilbert16-e-out-section-v1"
    assert payload["e_out_first_hit"] is True
    assert payload["outgoing_first_hit"] is False
    assert payload["hit"]["status"] == "certified"
    assert payload["hit"]["replayed"] is True
    assert payload["hit"]["transverse_negative"] is True
    kill_hits = payload["kill_hits"]
    assert set(kill_hits) == {"9/25", "1/16", "0"}
    assert all(row["status"] == "certified" and row["replayed"] for row in kill_hits.values())
    assert payload["grazing"]["status"] != "certified"
    honesty = payload["honesty"]
    assert honesty["e_out_first_hit"] is True
    assert honesty["outgoing_first_hit"] is False
    assert honesty["g1_passed"] is False
    assert honesty["full_hilbert16_solved"] is False
    reasons = honesty["_honesty_reasons"]
    assert reasons["g1_passed"]["reason"] == "gate_open"
    assert "e_out_first_hit" not in reasons


def test_e_out_section_refuses_grazing_e_sigma() -> None:
    grazing = certify_grazing_excluded(L=Fraction(0))
    assert grazing["status"] != "certified"
    far = certify_e_out(max_steps=8)
    assert far["status"] != "certified"
    assert KILL_L_PACK == (Fraction(9, 25), Fraction(1, 16), Fraction(0))


def test_local_e_out_section_never_earns_a_parent_flag() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_e_out_section")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["full_hilbert16_solved"] is False
    assert flags["hilbert16_part_b_quadratic_solved"] is False
