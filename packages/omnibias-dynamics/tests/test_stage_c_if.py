# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Kill-line Stage-C C=2 integrating factor; not T(h) orbit, first-hit, or G1."""

from __future__ import annotations

from fractions import Fraction

from omnibias.dynamics.hilbert16_ledger import default_h16_ledger, derived_parent_flags
from omnibias.dynamics.stage_c_if import (
    identity_verdicts,
    report,
    residual_c_triple,
    residual_sqrt_edge,
    residual_twice_six,
)


def test_stage_c_if_identities_are_exact() -> None:
    assert residual_c_triple() == 0
    assert residual_twice_six() == 0
    assert residual_sqrt_edge() == 0
    assert Fraction(2) * 3 == 6
    assert 12 * (Fraction(1, 4) - Fraction(1, 16)) == Fraction(9, 4)
    assert all(status == "PROVED" for status in identity_verdicts().values())


def test_stage_c_if_does_not_pass_g1() -> None:
    payload = report().to_payload()
    assert payload["schema"] == "hilbert16-stage-c-if-v1"
    assert payload["stage_c_if"] is True
    assert payload["sample_inside"] is True
    enclosure = payload["enclosure"]
    assert enclosure["below_declared"] is True
    assert enclosure["edge_below_cap"] is True
    assert enclosure["excludes_zero"] is True
    assert enclosure["factor_hi"] < 32.0
    assert enclosure["expo_hi"] < 4.0
    honesty = payload["honesty"]
    assert honesty["stage_c_if"] is True
    assert honesty["stage_c_env"] is False
    assert honesty["g1_passed"] is False
    assert honesty["full_hilbert16_solved"] is False
    reasons = honesty["_honesty_reasons"]
    assert reasons["g1_passed"]["reason"] == "gate_open"
    assert "stage_c_if" not in reasons


def test_local_stage_c_if_never_earns_a_parent_flag() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_stage_c_if")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["full_hilbert16_solved"] is False
    assert flags["hilbert16_part_b_quadratic_solved"] is False
