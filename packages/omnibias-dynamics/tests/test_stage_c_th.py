# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Kill-line Stage-C leading T_h; not an outgoing orbit, first-hit, or G1."""

from __future__ import annotations

from fractions import Fraction

from omnibias.dynamics.hilbert16_ledger import default_h16_ledger, derived_parent_flags
from omnibias.dynamics.stage_a import B_minus
from omnibias.dynamics.stage_c_th import (
    identity_verdicts,
    report,
    residual_mid_product,
    residual_th_half_room,
    residual_th_three_four,
)


def test_stage_c_th_identities_are_exact() -> None:
    assert residual_mid_product() == 0
    assert residual_mid_product(Fraction(1)) == 0
    assert residual_th_three_four() == 0
    assert residual_th_half_room() == 0
    assert B_minus(Fraction(1), Fraction(3, 5)) == Fraction(-9, 100)
    assert all(status == "PROVED" for status in identity_verdicts().values())


def test_stage_c_th_does_not_pass_g1() -> None:
    payload = report().to_payload()
    assert payload["schema"] == "hilbert16-stage-c-th-v1"
    assert payload["stage_c_th"] is True
    assert payload["sample_inside"] is True
    enclosure = payload["enclosure"]
    assert enclosure["below_declared"] is True
    assert enclosure["th_above_floor"] is True
    assert enclosure["excludes_zero"] is True
    assert enclosure["th_lo"] > 0.5
    assert enclosure["th_hi"] < 2.0
    honesty = payload["honesty"]
    assert honesty["stage_c_th"] is True
    assert honesty["stage_c_exit"] is False
    assert honesty["g1_passed"] is False
    assert honesty["full_hilbert16_solved"] is False
    reasons = honesty["_honesty_reasons"]
    assert reasons["g1_passed"]["reason"] == "gate_open"
    assert "stage_c_th" not in reasons


def test_local_stage_c_th_never_earns_a_parent_flag() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_stage_c_th")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["full_hilbert16_solved"] is False
    assert flags["hilbert16_part_b_quadratic_solved"] is False
