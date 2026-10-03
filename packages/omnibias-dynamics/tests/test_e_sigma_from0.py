# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Comparison GRAZING E_sigma from V=0; not Lohner or G1."""

from __future__ import annotations

from fractions import Fraction

from omnibias.dynamics.e_sigma_from0 import (
    KILL_L_PACK,
    enclose_from0,
    height_majorant,
    identity_verdicts,
    report,
    residual_de_sigma_lo,
    residual_f_cubic_factor,
    residual_log_taylor_num,
    residual_ratio_split,
    residual_vdot_rev_split,
)
from omnibias.dynamics.hilbert16_ledger import default_h16_ledger, derived_parent_flags
from omnibias.dynamics.vh_orbit import field_f


def test_e_sigma_from0_identities_are_exact() -> None:
    eps, v_coord, height = Fraction(1, 16), Fraction(1, 2), Fraction(1, 8)
    assert residual_f_cubic_factor(v_coord, eps) == 0
    assert residual_vdot_rev_split(v_coord, height, eps) == 0
    assert residual_ratio_split(v_coord, eps) == 0
    assert residual_log_taylor_num(Fraction(6, 79)) == 0
    assert residual_de_sigma_lo(
        Fraction(55, 79), Fraction(79, 80), Fraction(1, 4), Fraction(6, 5)
    ) == 0
    assert field_f(Fraction(0), Fraction(-2), eps, v_coord) == (
        (eps / 3) * v_coord * (v_coord * v_coord - 3 * v_coord - 6 * eps)
    )
    assert height_majorant(Fraction(6, 5)) > 0
    assert all(status == "PROVED" for status in identity_verdicts().values())
    assert KILL_L_PACK == (Fraction(9, 25), Fraction(1, 16), Fraction(0))
    collapsed = enclose_from0(eps=Fraction(0))
    assert collapsed["sign_change"] is False


def test_e_sigma_from0_hit_does_not_pass_g1() -> None:
    payload = report().to_payload()
    assert payload["schema"] == "hilbert16-e-sigma-from0-v1"
    assert payload["e_sigma_from0_hit"] is True
    assert payload["outgoing_first_hit"] is False
    assert set(payload["pack"]) == {"9/25", "1/16", "0"}
    assert all(
        row["sign_change"] and row["transverse"] for row in payload["pack"].values()
    )
    assert payload["refuse_v1"]["sign_change"] is False
    assert payload["lohner_from_start"]["status"] != "certified"
    honesty = payload["honesty"]
    assert honesty["e_sigma_from0_hit"] is True
    assert honesty["outgoing_first_hit"] is False
    assert honesty["g1_passed"] is False
    assert honesty["full_hilbert16_solved"] is False
    reasons = honesty["_honesty_reasons"]
    assert reasons["g1_passed"]["reason"] == "gate_open"
    assert "e_sigma_from0_hit" not in reasons


def test_local_e_sigma_from0_never_earns_a_parent_flag() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_e_sigma_from0")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["full_hilbert16_solved"] is False
    assert flags["hilbert16_part_b_quadratic_solved"] is False
