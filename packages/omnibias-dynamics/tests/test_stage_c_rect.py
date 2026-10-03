# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Kill-line Stage-C continuation rectangle; not first-hit or G1."""

from __future__ import annotations

from fractions import Fraction

from omnibias.dynamics.hilbert16_ledger import default_h16_ledger, derived_parent_flags
from omnibias.dynamics.stage_c_rect import (
    identity_verdicts,
    report,
    residual_amin_eps,
    residual_hmax_two,
    residual_twice_two,
)


def test_stage_c_rect_identities_are_exact() -> None:
    assert residual_hmax_two() == 0
    assert residual_twice_two() == 0
    assert residual_amin_eps() == 0
    assert Fraction(1) + Fraction(1) == Fraction(2)
    assert Fraction(1, 16) * Fraction(1, 4) == Fraction(1, 64)
    assert all(status == "PROVED" for status in identity_verdicts().values())


def test_stage_c_rect_does_not_pass_g1() -> None:
    payload = report().to_payload()
    assert payload["schema"] == "hilbert16-stage-c-rect-v1"
    assert payload["stage_c_rect"] is True
    assert payload["sample_inside"] is True
    enclosure = payload["enclosure"]
    assert enclosure["below_declared"] is True
    assert enclosure["boot_below"] is True
    assert enclosure["excludes_zero"] is True
    assert enclosure["v_left_hi"] < 3.0
    assert enclosure["v_right_hi"] < 0.03125
    honesty = payload["honesty"]
    assert honesty["stage_c_rect"] is True
    assert honesty["stage_c_boot"] is False
    assert honesty["g1_passed"] is False
    assert honesty["full_hilbert16_solved"] is False
    reasons = honesty["_honesty_reasons"]
    assert reasons["g1_passed"]["reason"] == "gate_open"
    assert "stage_c_rect" not in reasons


def test_local_stage_c_rect_never_earns_a_parent_flag() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_stage_c_rect")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["full_hilbert16_solved"] is False
    assert flags["hilbert16_part_b_quadratic_solved"] is False
