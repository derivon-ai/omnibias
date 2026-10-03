# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Alpha-0 T-h envelope after the x-corridor; not height first-hit or G1."""

from __future__ import annotations

from fractions import Fraction

from omnibias.dynamics.height_envelope import (
    identity_verdicts,
    q_envelope_ratio,
    report,
    residual_amgm_qbound,
    residual_envelope_exit,
    residual_q_envelope,
    residual_th_alpha0,
)
from omnibias.dynamics.hilbert16_ledger import default_h16_ledger, derived_parent_flags


def test_height_envelope_identities_are_exact() -> None:
    r1, r2, xstar = Fraction(1, 5), Fraction(4), Fraction(1)
    eps, y0, hmax = Fraction(1, 16), Fraction(6), Fraction(1)
    t_exit = (eps * xstar) ** 2 / 2
    h_exit = eps**3 * y0
    q_abs = eps**3 * (xstar - r1) * (r2 - xstar)
    t_alpha0 = t_exit + hmax - h_exit
    assert residual_amgm_qbound(eps, 2 * eps) == 0
    assert residual_envelope_exit(t_exit, eps, h_exit, xstar, y0) == 0
    assert residual_th_alpha0(t_alpha0, hmax, t_exit, h_exit) == 0
    assert residual_q_envelope(q_abs, eps, t_exit, xstar, r1, r2) == 0
    assert t_exit > h_exit
    assert 0 < t_exit - h_exit < Fraction(1, 8)
    assert q_envelope_ratio(xstar, r1, r2) == Fraction(8, 5)
    assert all(status == "PROVED" for status in identity_verdicts().values())


def test_height_envelope_does_not_pass_g1() -> None:
    payload = report().to_payload()
    assert payload["schema"] == "hilbert16-height-envelope-v1"
    assert payload["height_envelope_alpha0"] is True
    assert payload["outgoing_first_hit"] is False
    assert payload["te_above_he"] is True
    assert payload["gap_below_margin"] is True
    honesty = payload["honesty"]
    assert honesty["height_envelope_alpha0"] is True
    assert honesty["outgoing_first_hit"] is False
    assert honesty["g1_passed"] is False
    assert honesty["full_hilbert16_solved"] is False
    reasons = honesty["_honesty_reasons"]
    assert reasons["g1_passed"]["reason"] == "gate_open"
    assert reasons["outgoing_first_hit"]["reason"] == "unimplemented"
    assert "height_envelope_alpha0" not in reasons


def test_local_height_envelope_never_earns_a_parent_flag() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_height_envelope")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["full_hilbert16_solved"] is False
    assert flags["hilbert16_part_b_quadratic_solved"] is False
