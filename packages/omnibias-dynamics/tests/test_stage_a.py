# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Kill-line Stage-A wall enclosure; not dx_e/dkappa, Stage C, or G1."""

from __future__ import annotations

from fractions import Fraction

from omnibias.dynamics.hilbert16_ledger import default_h16_ledger, derived_parent_flags
from omnibias.dynamics.stage_a import (
    B_minus,
    identity_verdicts,
    r2_kill,
    report,
    residual_a_min,
    residual_B_slope,
    residual_B_wall,
    wall_left,
    wall_right,
)
from omnibias.dynamics.stage_b import r1_kill


def test_stage_a_identities_are_exact() -> None:
    sep = Fraction(3, 5)
    assert residual_B_wall(sep) == 0
    assert residual_B_slope(sep) == 0
    assert residual_a_min() == 0
    assert r1_kill(sep) == Fraction(7, 10)
    assert r2_kill(sep) == Fraction(13, 10)
    assert wall_left(sep) == Fraction(5, 8)
    assert wall_right(sep) == Fraction(31, 40)
    assert B_minus(wall_left(sep), sep) == Fraction(81, 1600)
    assert all(status == "PROVED" for status in identity_verdicts().values())


def test_stage_a_wall_does_not_pass_g1() -> None:
    payload = report().to_payload()
    assert payload["schema"] == "hilbert16-stage-a-v1"
    assert payload["stage_a_wall"] is True
    assert payload["sample_inside"] is True
    enclosure = payload["enclosure"]
    assert enclosure["below_declared"] is True
    assert enclosure["psi_below_declared"] is True
    assert enclosure["excludes_zero"] is True
    assert enclosure["slope_ok"] is True
    assert enclosure["a_lo"] > 0.25
    assert enclosure["psi_factor_hi"] < 0.25
    honesty = payload["honesty"]
    assert honesty["stage_a_wall"] is True
    assert honesty["orbit_continuation_on_kill"] is False
    assert honesty["g1_passed"] is False
    assert honesty["full_hilbert16_solved"] is False
    reasons = honesty["_honesty_reasons"]
    assert reasons["g1_passed"]["reason"] == "gate_open"
    assert "stage_a_wall" not in reasons


def test_local_stage_a_never_earns_a_parent_flag() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_stage_a")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["full_hilbert16_solved"] is False
    assert flags["hilbert16_part_b_quadratic_solved"] is False
