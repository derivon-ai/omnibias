# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Restored (V,h) hypotheses after the x-corridor; not height first-hit or G1."""

from __future__ import annotations

from fractions import Fraction

from omnibias.core.verified.interval import Interval
from omnibias.dynamics.hilbert16_ledger import default_h16_ledger, derived_parent_flags
from omnibias.dynamics.post_corridor import (
    identity_verdicts,
    integrating_factor_exponent_majorant,
    report,
    residual_h_exit,
    residual_q_leading,
    residual_T_kinetic,
    residual_V_embed,
    th_leading_value,
)


def test_post_corridor_identities_are_exact() -> None:
    r1, r2 = Fraction(1, 5), Fraction(4)
    lam1, L = -(r1 + r2), r1 * r2
    xstar, eps, y0 = Fraction(1), Fraction(1, 7), Fraction(8)
    v_star = -eps * xstar
    t_star = (eps * xstar) ** 2 / 2
    h_e = eps**3 * y0
    assert residual_V_embed(v_star, eps, xstar) == 0
    assert residual_T_kinetic(t_star, eps, xstar) == 0
    assert residual_h_exit(h_e, eps, y0) == 0
    assert residual_q_leading(L, lam1, xstar, r1, r2) == 0
    assert th_leading_value(y0, xstar, r1, r2) > y0 / 2
    assert xstar > r1
    assert all(status == "PROVED" for status in identity_verdicts().values())


def test_integrating_factor_majorant_vanishes_and_does_not_pass_g1() -> None:
    bound = integrating_factor_exponent_majorant(Interval.from_rational(Fraction(1, 100)))
    assert bound.hi < 0.6
    payload = report().to_payload()
    assert payload["schema"] == "hilbert16-post-corridor-v1"
    assert payload["post_corridor_margin"] is True
    assert payload["outgoing_first_hit"] is False
    assert payload["a_wall_failed"] is True
    assert payload["restored_x_margin"] is True
    assert payload["th_leading_half"] is True
    honesty = payload["honesty"]
    assert honesty["post_corridor_margin"] is True
    assert honesty["outgoing_first_hit"] is False
    assert honesty["g1_passed"] is False
    assert honesty["full_hilbert16_solved"] is False
    reasons = honesty["_honesty_reasons"]
    assert reasons["g1_passed"]["reason"] == "gate_open"
    assert reasons["outgoing_first_hit"]["reason"] == "unimplemented"
    assert "post_corridor_margin" not in reasons


def test_local_post_corridor_never_earns_a_parent_flag() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_post_corridor")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["full_hilbert16_solved"] is False
    assert flags["hilbert16_part_b_quadratic_solved"] is False
