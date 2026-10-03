# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Kill-line Stage-C C=2 T-h bootstrap; not first-hit or G1."""

from __future__ import annotations

from fractions import Fraction

from omnibias.dynamics.hilbert16_ledger import default_h16_ledger, derived_parent_flags
from omnibias.dynamics.stage_c_boot import (
    identity_verdicts,
    report,
    residual_c_log,
    residual_lin_fifteen,
    residual_one_k,
)


def test_stage_c_boot_identities_are_exact() -> None:
    assert residual_one_k() == 0
    assert residual_lin_fifteen() == 0
    assert residual_c_log() == 0
    assert Fraction(1) + Fraction(6) == Fraction(7)
    assert Fraction(2) * Fraction(7) * Fraction(3) == Fraction(42)
    assert all(status == "PROVED" for status in identity_verdicts().values())


def test_stage_c_boot_does_not_pass_g1() -> None:
    payload = report().to_payload()
    assert payload["schema"] == "hilbert16-stage-c-boot-v1"
    assert payload["stage_c_boot"] is True
    assert payload["sample_inside"] is True
    enclosure = payload["enclosure"]
    assert enclosure["below_declared"] is True
    assert enclosure["k_below"] is True
    assert enclosure["excludes_zero"] is True
    assert enclosure["total_hi"] < 1.0
    assert enclosure["lin_hi"] < 1.0
    honesty = payload["honesty"]
    assert honesty["stage_c_boot"] is True
    assert honesty["stage_c_k"] is False
    assert honesty["g1_passed"] is False
    assert honesty["full_hilbert16_solved"] is False
    reasons = honesty["_honesty_reasons"]
    assert reasons["g1_passed"]["reason"] == "gate_open"
    assert "stage_c_boot" not in reasons


def test_local_stage_c_boot_never_earns_a_parent_flag() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_stage_c_boot")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["full_hilbert16_solved"] is False
    assert flags["hilbert16_part_b_quadratic_solved"] is False
