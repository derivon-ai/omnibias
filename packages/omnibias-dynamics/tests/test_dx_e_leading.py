# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Kill-line dx_e leading factors; not uniform-in-chi, Stage C, or G1."""

from __future__ import annotations

from fractions import Fraction

from omnibias.dynamics.dx_e_leading import (
    identity_verdicts,
    report,
    residual_net_floor,
    residual_prefactor,
    residual_slope_half,
)
from omnibias.dynamics.hilbert16_ledger import default_h16_ledger, derived_parent_flags
from omnibias.dynamics.stage_a import wall_left


def test_dx_e_leading_identities_are_exact() -> None:
    assert residual_prefactor() == 0
    assert residual_slope_half() == 0
    assert residual_net_floor() == 0
    assert wall_left(Fraction(3, 5)) == Fraction(5, 8)
    assert all(status == "PROVED" for status in identity_verdicts().values())


def test_dx_e_leading_does_not_pass_g1() -> None:
    payload = report().to_payload()
    assert payload["schema"] == "hilbert16-dx-e-leading-v1"
    assert payload["dx_e_leading"] is True
    assert payload["sample_inside"] is True
    enclosure = payload["enclosure"]
    assert enclosure["below_declared"] is True
    assert enclosure["net_below_declared"] is True
    assert enclosure["excludes_zero"] is True
    assert enclosure["pref_hi"] < 0.5
    assert enclosure["after_lo"] > 0.125
    honesty = payload["honesty"]
    assert honesty["dx_e_leading"] is True
    assert honesty["chi_b_bound"] is False
    assert honesty["g1_passed"] is False
    assert honesty["full_hilbert16_solved"] is False
    reasons = honesty["_honesty_reasons"]
    assert reasons["g1_passed"]["reason"] == "gate_open"
    assert "dx_e_leading" not in reasons


def test_local_dx_e_leading_never_earns_a_parent_flag() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_dx_e_leading")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["full_hilbert16_solved"] is False
    assert flags["hilbert16_part_b_quadratic_solved"] is False
