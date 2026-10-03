# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Kill-line Stage-C comparison first-hit of h=1; not Lohner or G1."""

from __future__ import annotations

from fractions import Fraction

from omnibias.dynamics.hilbert16_ledger import default_h16_ledger, derived_parent_flags
from omnibias.dynamics.stage_c_hit import (
    identity_verdicts,
    report,
    residual_hmax_gap,
    residual_inv_floor,
    residual_time_pre,
)


def test_stage_c_hit_identities_are_exact() -> None:
    assert residual_inv_floor() == 0
    assert residual_time_pre() == 0
    assert residual_hmax_gap() == 0
    assert Fraction(1) / Fraction(1, 64) == Fraction(64)
    assert Fraction(64) * Fraction(3) == Fraction(192)
    assert Fraction(1) - Fraction(1, 4096) == Fraction(4095, 4096)
    assert all(status == "PROVED" for status in identity_verdicts().values())


def test_stage_c_hit_does_not_pass_g1() -> None:
    payload = report().to_payload()
    assert payload["schema"] == "hilbert16-stage-c-hit-v1"
    assert payload["stage_c_hit"] is True
    assert payload["sample_inside"] is True
    enclosure = payload["enclosure"]
    assert enclosure["below_declared"] is True
    assert enclosure["h1_below"] is True
    assert enclosure["excludes_zero"] is True
    assert enclosure["time_hi"] < 1024.0
    assert enclosure["pre_hi"] < 256.0
    honesty = payload["honesty"]
    assert honesty["stage_c_hit"] is True
    assert honesty["stage_c_rect"] is False
    assert honesty["g1_passed"] is False
    assert honesty["full_hilbert16_solved"] is False
    reasons = honesty["_honesty_reasons"]
    assert reasons["g1_passed"]["reason"] == "gate_open"
    assert "stage_c_hit" not in reasons


def test_local_stage_c_hit_never_earns_a_parent_flag() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_stage_c_hit")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["full_hilbert16_solved"] is False
    assert flags["hilbert16_part_b_quadratic_solved"] is False
