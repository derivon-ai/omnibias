# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Bounded x-corridor I-map; not height-section first-hit or G1."""

from __future__ import annotations

from fractions import Fraction

from omnibias.core.verified.interval import Interval
from omnibias.dynamics.hilbert16_ledger import default_h16_ledger, derived_parent_flags
from omnibias.dynamics.outgoing_corridor import (
    a_wall_gap,
    identity_verdicts,
    r1_log_majorant,
    report,
    residual_interface_embed,
    residual_J_numerator,
    residual_root_sum,
)


def test_corridor_identities_are_exact() -> None:
    r1, r2, lam1 = Fraction(1, 5), Fraction(4), Fraction(-21, 5)
    theta, eps = Fraction(1, 2), Fraction(1, 7)
    x_lo = r1 * (1 + theta)
    assert residual_J_numerator(Fraction(1), r1, r2) == 0
    assert residual_root_sum(r1, r2, lam1) == 0
    assert residual_interface_embed(-eps * x_lo, eps, r1, theta, x_lo) == 0
    assert a_wall_gap(r1, Fraction(1, 4)) < 0
    assert all(status == "PROVED" for status in identity_verdicts().values())


def test_r1_log_majorant_vanishes_and_does_not_pass_g1() -> None:
    bound = r1_log_majorant(Interval.from_rational(Fraction(1, 10000)))
    assert bound.lo >= -1e-12
    assert bound.hi < 0.03
    payload = report().to_payload()
    assert payload["schema"] == "hilbert16-outgoing-corridor-v1"
    assert payload["outgoing_x_corridor_bounded"] is True
    assert payload["outgoing_first_hit"] is False
    assert payload["a_wall_failed"] is True
    honesty = payload["honesty"]
    assert honesty["outgoing_x_corridor_bounded"] is True
    assert honesty["outgoing_first_hit"] is False
    assert honesty["g1_passed"] is False
    assert honesty["full_hilbert16_solved"] is False
    reasons = honesty["_honesty_reasons"]
    assert reasons["g1_passed"]["reason"] == "gate_open"
    assert reasons["outgoing_first_hit"]["reason"] == "unimplemented"
    assert "outgoing_x_corridor_bounded" not in reasons


def test_local_outgoing_corridor_never_earns_a_parent_flag() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_outgoing_corridor")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["full_hilbert16_solved"] is False
    assert flags["hilbert16_part_b_quadratic_solved"] is False
