# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Exact slow-line entry-exit identities; not a C2 or G1 certificate."""

from __future__ import annotations

from fractions import Fraction

from omnibias.dynamics.entry_exit_leading import (
    height_inflation_delta_x,
    identity_verdicts,
    kill_tracked_log,
    report,
    residual_double_root_entry_exit,
    residual_height_inflation_dx_dy,
    residual_partial_fraction_numerators,
    residual_product_alpha_one,
    tracked_product_identity_log,
)
from omnibias.dynamics.scale_dichotomy import KILL_EPS, kill_sep, rstar


def test_partial_fraction_and_double_root_are_exact() -> None:
    assert residual_partial_fraction_numerators(Fraction(2), Fraction(1), Fraction(4)) == 0
    assert residual_double_root_entry_exit(Fraction(5, 2), Fraction(3, 2)) == 0
    assert residual_product_alpha_one(Fraction(2), Fraction(3), Fraction(1, 2), Fraction(5)) == 0
    assert residual_height_inflation_dx_dy(
        Fraction(1, 3), Fraction(7, 2), Fraction(5), Fraction(4)
    ) == 0
    assert all(status == "PROVED" for status in identity_verdicts().values())


def test_tracked_product_log_identity_and_kill_sequence_is_negative() -> None:
    eps = KILL_EPS
    left, right = tracked_product_identity_log(0.5, eps**3, eps, 1.0, 2.0 * eps)
    assert abs(left - right) < 1e-12
    kill_log = kill_tracked_log(eps=eps, c_factor=2.0)
    assert kill_log < 0.0
    assert abs(kill_log - (2.0 - 4.0 * eps) * (-1.0 / (eps * eps))) < 1e-12


def test_height_inflation_is_order_epsilon_on_the_kill_sequence() -> None:
    eps = KILL_EPS
    sep = kill_sep(eps)
    wall = rstar(-3.0)
    delta = height_inflation_delta_x(eps, 1.0, wall, sep * sep, 1.0)
    assert delta < 2.0 * eps / wall
    assert delta > 0.0


def test_report_does_not_pass_g1() -> None:
    payload = report().to_payload()
    assert payload["kill_log_negative"] is True
    assert payload["product_bounded_on_unit_sep"] is True
    assert all(status == "PROVED" for status in payload["identities"].values())
    honesty = payload["honesty"]
    assert honesty["g1_passed"] is False
    assert honesty["g4_passed"] is False
    assert honesty["full_hilbert16_solved"] is False
    assert honesty["physical_return_membership_proved"] is False
    reasons = honesty["_honesty_reasons"]
    assert reasons["g1_passed"]["reason"] == "gate_open"
