# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Kill-line Stage-C C=2 tight T(h) ratio; not T-h=O(eps) or G1."""

from __future__ import annotations

from fractions import Fraction

from omnibias.dynamics.hilbert16_ledger import default_h16_ledger, derived_parent_flags
from omnibias.dynamics.stage_c_k import (
    identity_verdicts,
    report,
    residual_k_slope,
    residual_six_eps,
    residual_twice_three,
)


def test_stage_c_k_identities_are_exact() -> None:
    assert residual_six_eps() == 0
    assert residual_twice_three() == 0
    assert residual_k_slope() == 0
    assert Fraction(6) * Fraction(1, 16) == Fraction(3, 8)
    assert Fraction(6) - Fraction(16, 7) == Fraction(26, 7)
    assert all(status == "PROVED" for status in identity_verdicts().values())


def test_stage_c_k_does_not_pass_g1() -> None:
    payload = report().to_payload()
    assert payload["schema"] == "hilbert16-stage-c-k-v1"
    assert payload["stage_c_k"] is True
    assert payload["sample_inside"] is True
    enclosure = payload["enclosure"]
    assert enclosure["below_declared"] is True
    assert enclosure["pref_below"] is True
    assert enclosure["excludes_zero"] is True
    assert enclosure["k_hi"] < 6.0
    assert enclosure["factor_hi"] < 3.0
    honesty = payload["honesty"]
    assert honesty["stage_c_k"] is True
    assert honesty["stage_c_lo"] is False
    assert honesty["g1_passed"] is False
    assert honesty["full_hilbert16_solved"] is False
    reasons = honesty["_honesty_reasons"]
    assert reasons["g1_passed"]["reason"] == "gate_open"
    assert "stage_c_k" not in reasons


def test_local_stage_c_k_never_earns_a_parent_flag() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_stage_c_k")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["full_hilbert16_solved"] is False
    assert flags["hilbert16_part_b_quadratic_solved"] is False
