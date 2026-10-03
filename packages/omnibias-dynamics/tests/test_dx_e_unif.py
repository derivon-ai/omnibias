# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Kill-line uniform-in-chi dx_e; not Stage C, first-hit, or G1."""

from __future__ import annotations

from fractions import Fraction

from omnibias.dynamics.dx_e_unif import (
    identity_verdicts,
    report,
    residual_c_weaker,
    residual_extra_half,
    residual_lift_nine,
)
from omnibias.dynamics.hilbert16_ledger import default_h16_ledger, derived_parent_flags
from omnibias.dynamics.stage_b import r1_kill


def test_dx_e_unif_identities_are_exact() -> None:
    assert residual_extra_half() == 0
    assert residual_c_weaker() == 0
    assert residual_lift_nine() == 0
    assert r1_kill(Fraction(3, 5)) == Fraction(7, 10)
    assert all(status == "PROVED" for status in identity_verdicts().values())


def test_dx_e_unif_does_not_pass_g1() -> None:
    payload = report().to_payload()
    assert payload["schema"] == "hilbert16-dx-e-unif-v1"
    assert payload["dx_e_unif"] is True
    assert payload["sample_inside"] is True
    enclosure = payload["enclosure"]
    assert enclosure["below_declared"] is True
    assert enclosure["extra_above_floor"] is True
    assert enclosure["excludes_zero"] is True
    assert enclosure["c_hi"] < 2.0
    assert enclosure["extra_lo"] > 0.0625
    honesty = payload["honesty"]
    assert honesty["dx_e_unif"] is True
    assert honesty["dx_e_leading"] is False
    assert honesty["g1_passed"] is False
    assert honesty["full_hilbert16_solved"] is False
    reasons = honesty["_honesty_reasons"]
    assert reasons["g1_passed"]["reason"] == "gate_open"
    assert "dx_e_unif" not in reasons


def test_local_dx_e_unif_never_earns_a_parent_flag() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_dx_e_unif")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["full_hilbert16_solved"] is False
    assert flags["hilbert16_part_b_quadratic_solved"] is False
