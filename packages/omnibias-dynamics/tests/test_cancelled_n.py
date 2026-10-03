# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Cancelled-N holomorphic Z on the kill compact; not first-hit or G1."""

from __future__ import annotations

from fractions import Fraction

from omnibias.dynamics.cancelled_n import (
    enclose_holomorphic_z,
    identity_verdicts,
    report,
    residual_delta_slow,
    residual_ell_V_nu,
    residual_n_cancelled,
    residual_t_L,
    residual_t_lambda,
    residual_z0_holomorphic,
)
from omnibias.dynamics.hilbert16_ledger import default_h16_ledger, derived_parent_flags


def test_cancelled_n_identities_are_exact() -> None:
    nu, v0, v, kay = Fraction(5, 16), Fraction(4, 5), Fraction(1, 2), Fraction(2)
    lam0, lam1 = Fraction(0), Fraction(288, 125)
    assert residual_delta_slow(nu, v, v0, kay, lam0, lam1) == 0
    assert residual_ell_V_nu(nu, v, v0) == 0
    assert residual_t_lambda(nu, v, v0, kay, lam1) == 0
    assert residual_t_L(nu, v, v0, kay, lam0) == 0
    assert residual_n_cancelled(nu, v, v0, kay, lam0, lam1) == 0
    assert residual_z0_holomorphic(nu, v, v0) == 0
    assert all(status == "PROVED" for status in identity_verdicts().values())


def test_cancelled_n_usable_z_does_not_pass_g1() -> None:
    payload = report().to_payload()
    assert payload["schema"] == "hilbert16-cancelled-n-v1"
    assert payload["enclosure"]["finite"] is True
    assert payload["enclosure"]["picard_included"] is True
    assert payload["enclosure"]["usable_c2_delta"] is True
    assert payload["enclosure"]["remainder_majorant"] < 1.0
    assert payload["sample_below_bound"] is True
    honesty = payload["honesty"]
    assert honesty["cancelled_n_usable_z"] is True
    assert honesty["kill_z_compact_bound"] is False
    assert honesty["outgoing_first_hit"] is False
    assert honesty["g1_passed"] is False
    assert honesty["full_hilbert16_solved"] is False
    reasons = honesty["_honesty_reasons"]
    assert reasons["g1_passed"]["reason"] == "gate_open"
    assert reasons["outgoing_first_hit"]["reason"] == "unimplemented"
    assert "cancelled_n_usable_z" not in reasons


def test_larger_cancelled_n_nu_box_refuses_picard() -> None:
    wide = enclose_holomorphic_z(nu_hi=0.2)
    assert wide["picard_included"] is False
    assert wide["usable_c2_delta"] is False


def test_local_cancelled_n_never_earns_a_parent_flag() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_cancelled_n")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["full_hilbert16_solved"] is False
    assert flags["hilbert16_part_b_quadratic_solved"] is False
