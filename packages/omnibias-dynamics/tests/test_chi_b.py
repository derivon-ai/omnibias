# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Kill-line chi_b threshold; not dx_e/dkappa, Stage C, or G1."""

from __future__ import annotations

from fractions import Fraction

from omnibias.dynamics.chi_b import (
    identity_verdicts,
    report,
    residual_c_decay,
    residual_chi_declared,
    residual_limit_nine,
)
from omnibias.dynamics.hilbert16_ledger import default_h16_ledger, derived_parent_flags


def test_chi_b_identities_are_exact() -> None:
    assert residual_limit_nine() == 0
    assert residual_c_decay() == 0
    assert residual_chi_declared() == 0
    assert (1 + Fraction(1, 8)) / Fraction(1, 8) == 9
    assert all(status == "PROVED" for status in identity_verdicts().values())


def test_chi_b_bound_does_not_pass_g1() -> None:
    payload = report().to_payload()
    assert payload["schema"] == "hilbert16-chi-b-v1"
    assert payload["chi_b_bound"] is True
    assert payload["sample_inside"] is True
    enclosure = payload["enclosure"]
    assert enclosure["below_declared"] is True
    assert enclosure["chi_below_declared"] is True
    assert enclosure["excludes_zero"] is True
    assert enclosure["s_hi"] < 3.0
    assert enclosure["chi_hi"] < 9.0
    honesty = payload["honesty"]
    assert honesty["chi_b_bound"] is True
    assert honesty["stage_a_wall"] is False
    assert honesty["g1_passed"] is False
    assert honesty["full_hilbert16_solved"] is False
    reasons = honesty["_honesty_reasons"]
    assert reasons["g1_passed"]["reason"] == "gate_open"
    assert "chi_b_bound" not in reasons


def test_local_chi_b_never_earns_a_parent_flag() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_chi_b")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["full_hilbert16_solved"] is False
    assert flags["hilbert16_part_b_quadratic_solved"] is False
