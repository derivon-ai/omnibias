# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Kill-line Stage-C comparison first-hit of E_out; not Lohner or G1."""

from __future__ import annotations

from fractions import Fraction

from omnibias.dynamics.hilbert16_ledger import default_h16_ledger, derived_parent_flags
from omnibias.dynamics.stage_c_sec import (
    identity_verdicts,
    report,
    residual_corr_sum,
    residual_rho_start,
    residual_vh_floor,
)


def test_stage_c_sec_identities_are_exact() -> None:
    assert residual_rho_start() == 0
    assert residual_vh_floor() == 0
    assert residual_corr_sum() == 0
    assert Fraction(1, 4) - Fraction(1, 12) == Fraction(1, 6)
    assert Fraction(1, 2) / Fraction(2) == Fraction(1, 4)
    assert Fraction(1, 64) + Fraction(1, 256) == Fraction(5, 256)
    assert all(status == "PROVED" for status in identity_verdicts().values())


def test_stage_c_sec_does_not_pass_g1() -> None:
    payload = report().to_payload()
    assert payload["schema"] == "hilbert16-stage-c-sec-v1"
    assert payload["stage_c_sec"] is True
    assert payload["sample_inside"] is True
    enclosure = payload["enclosure"]
    assert enclosure["below_declared"] is True
    assert enclosure["start_positive"] is True
    assert enclosure["transverse"] is True
    assert enclosure["before_hmax"] is True
    assert enclosure["e_lo"] > 0.0
    assert enclosure["room_lo"] > 0.0
    assert enclosure["hhit_hi"] < 0.125
    honesty = payload["honesty"]
    assert honesty["stage_c_sec"] is True
    assert honesty["stage_c_hit"] is False
    assert honesty["g1_passed"] is False
    assert honesty["full_hilbert16_solved"] is False
    reasons = honesty["_honesty_reasons"]
    assert reasons["g1_passed"]["reason"] == "gate_open"
    assert "stage_c_sec" not in reasons


def test_local_stage_c_sec_never_earns_a_parent_flag() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_stage_c_sec")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["full_hilbert16_solved"] is False
    assert flags["hilbert16_part_b_quadratic_solved"] is False
