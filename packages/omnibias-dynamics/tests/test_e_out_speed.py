# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Kill-line comparison speed bound; not Lohner-for-every-eps, GRAZING, or G1."""

from __future__ import annotations

from fractions import Fraction

from omnibias.dynamics.e_out_speed import (
    cmp_F,
    enclose_g,
    enclose_phi,
    identity_verdicts,
    report,
    residual_F_expand,
    residual_F_V_phi,
    residual_phi_right,
    residual_T_cubic,
    t_cubic_majorant,
)
from omnibias.dynamics.hilbert16_ledger import default_h16_ledger, derived_parent_flags
from omnibias.dynamics.vh_orbit import field_f, field_g


def test_e_out_speed_identities_are_exact() -> None:
    eps = Fraction(1, 16)
    v_coord = Fraction(-1, 8)
    rho = Fraction(1, 4)
    assert residual_F_expand(v_coord, eps) == 0
    assert residual_F_V_phi(v_coord, eps) == 0
    assert residual_phi_right(eps) == 0
    assert residual_T_cubic(eps, rho, t_cubic_majorant(eps, rho)) == 0
    assert t_cubic_majorant(eps, rho) == Fraction(256)
    expanded = (
        field_f(Fraction(0), Fraction(-2), eps, v_coord)
        + (4 * eps**3) * field_g(eps, v_coord)
    )
    assert cmp_F(v_coord, eps) == expanded
    assert all(status == "PROVED" for status in identity_verdicts().values())
    phi_box = enclose_phi()
    g_box = enclose_g()
    assert phi_box["strictly_positive"] is True
    assert g_box["strictly_negative"] is True
    origin = enclose_phi(eps=Fraction(0), v_lo=Fraction(0), v_hi=Fraction(0))
    assert origin["strictly_positive"] is False


def test_e_out_speed_bound_does_not_pass_g1() -> None:
    payload = report().to_payload()
    assert payload["schema"] == "hilbert16-e-out-speed-v1"
    assert payload["e_out_speed_bound"] is True
    assert payload["outgoing_first_hit"] is False
    honesty = payload["honesty"]
    assert honesty["e_out_speed_bound"] is True
    assert honesty["outgoing_first_hit"] is False
    assert honesty["g1_passed"] is False
    assert honesty["full_hilbert16_solved"] is False
    reasons = honesty["_honesty_reasons"]
    assert reasons["g1_passed"]["reason"] == "gate_open"
    assert "e_out_speed_bound" not in reasons


def test_local_e_out_speed_never_earns_a_parent_flag() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_e_out_speed")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["full_hilbert16_solved"] is False
    assert flags["hilbert16_part_b_quadratic_solved"] is False
