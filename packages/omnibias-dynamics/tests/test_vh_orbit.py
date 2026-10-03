# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Cubic (V,h) Lohner prefix and V=-1/4 first-hit; not physical E_sigma or G1."""

from __future__ import annotations

from fractions import Fraction

from omnibias.dynamics.hilbert16_ledger import default_h16_ledger, derived_parent_flags
from omnibias.dynamics.vh_orbit import (
    certify_vwall,
    enclose_lohner_prefix,
    field_f,
    field_g,
    identity_verdicts,
    report,
    residual_g_jet_vh,
    residual_hdot_vh,
    residual_q_plus_f,
    residual_Vdot_split,
)


def test_vh_orbit_identities_are_exact() -> None:
    L, lam1, eps = Fraction(9, 25), Fraction(-2), Fraction(1, 16)
    nu, v_coord = eps, -eps
    height = 4 * eps**3
    w = eps
    eff = field_f(L, lam1, eps, v_coord)
    gee = field_g(nu, v_coord)
    assert residual_q_plus_f(L, lam1, eps, w) == 0
    assert residual_hdot_vh(-v_coord * height, v_coord, height) == 0
    assert residual_g_jet_vh(gee, nu, v_coord) == 0
    assert residual_Vdot_split(eff + height * gee, eff, height, gee) == 0
    assert all(status == "PROVED" for status in identity_verdicts().values())


def test_vh_orbit_certified_does_not_pass_g1() -> None:
    payload = report().to_payload()
    assert payload["schema"] == "hilbert16-vh-orbit-v1"
    assert payload["vh_orbit_certified"] is True
    assert payload["outgoing_first_hit"] is False
    assert payload["hit"]["status"] == "certified"
    assert payload["hit"]["replayed"] is True
    assert payload["lohner"]["v_negative"] is True
    assert payload["lohner"]["below_margin"] is True
    honesty = payload["honesty"]
    assert honesty["vh_orbit_certified"] is True
    assert honesty["outgoing_first_hit"] is False
    assert honesty["g1_passed"] is False
    assert honesty["full_hilbert16_solved"] is False
    reasons = honesty["_honesty_reasons"]
    assert reasons["g1_passed"]["reason"] == "gate_open"
    assert "vh_orbit_certified" not in reasons


def test_vh_orbit_refuses_a_far_wall() -> None:
    far = certify_vwall(wall=Fraction(2), max_steps=16)
    assert far["status"] != "certified"
    blown = enclose_lohner_prefix(step=5.0, n_steps=12)
    assert blown["v_negative"] is False
    assert blown["below_margin"] is False


def test_local_vh_orbit_never_earns_a_parent_flag() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_vh_orbit")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["full_hilbert16_solved"] is False
    assert flags["hilbert16_part_b_quadratic_solved"] is False
