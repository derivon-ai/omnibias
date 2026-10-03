# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Finite shrinking-eps E_out pack; not uniform eps->0, GRAZING, or G1."""

from __future__ import annotations

from fractions import Fraction

from omnibias.dynamics.e_out_eps import (
    EPS_PACK,
    horizon_majorant,
    identity_verdicts,
    report,
    residual_h0_matching,
    residual_horizon_n2,
    residual_v0_matching,
    residual_vdot_kill_init,
)
from omnibias.dynamics.hilbert16_ledger import default_h16_ledger, derived_parent_flags
from omnibias.dynamics.vh_orbit import field_f, field_g


def test_e_out_eps_identities_are_exact() -> None:
    eps = Fraction(1, 16)
    n = Fraction(16)
    v0 = -eps
    h0 = 4 * eps**3
    vdot = field_f(Fraction(0), Fraction(-2), eps, v0) + h0 * field_g(eps, v0)
    assert residual_v0_matching(v0, eps) == 0
    assert residual_h0_matching(h0, eps) == 0
    assert residual_horizon_n2(n, horizon_majorant(n)) == 0
    assert residual_vdot_kill_init(vdot, eps) == 0
    assert all(status == "PROVED" for status in identity_verdicts().values())
    assert EPS_PACK == (16, 20, 25)


def test_e_out_eps_pack_does_not_pass_g1() -> None:
    payload = report().to_payload()
    assert payload["schema"] == "hilbert16-e-out-eps-v1"
    assert payload["e_out_eps_pack"] is True
    assert payload["outgoing_first_hit"] is False
    assert set(payload["pack"]) == {"16", "20", "25"}
    assert all(row["status"] == "certified" and row["replayed"] for row in payload["pack"].values())
    assert payload["short"]["status"] != "certified"
    honesty = payload["honesty"]
    assert honesty["e_out_eps_pack"] is True
    assert honesty["outgoing_first_hit"] is False
    assert honesty["g1_passed"] is False
    assert honesty["full_hilbert16_solved"] is False
    reasons = honesty["_honesty_reasons"]
    assert reasons["g1_passed"]["reason"] == "gate_open"
    assert "e_out_eps_pack" not in reasons


def test_local_e_out_eps_never_earns_a_parent_flag() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_e_out_eps")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["full_hilbert16_solved"] is False
    assert flags["hilbert16_part_b_quadratic_solved"] is False
