# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Kill-line Stage-B Picard enclosure; not Stage A/C, C2, first-hit, or G1."""

from __future__ import annotations

from fractions import Fraction

from omnibias.dynamics.hilbert16_ledger import default_h16_ledger, derived_parent_flags
from omnibias.dynamics.stage_b import (
    identity_verdicts,
    r1_kill,
    report,
    residual_alpha_pos,
    residual_dx_declared,
    residual_r1_half,
)


def test_stage_b_identities_are_exact() -> None:
    sep = Fraction(3, 5)
    assert residual_r1_half(sep) == 0
    assert residual_dx_declared() == 0
    assert residual_alpha_pos() == 0
    assert r1_kill(sep) == Fraction(7, 10)
    assert all(status == "PROVED" for status in identity_verdicts().values())


def test_stage_b_dx_does_not_pass_g1() -> None:
    payload = report().to_payload()
    assert payload["schema"] == "hilbert16-stage-b-v1"
    assert payload["stage_b_dx"] is True
    assert payload["orbit_continuation_on_kill"] is True
    assert payload["sample_inside"] is True
    enclosure = payload["enclosure"]
    assert enclosure["picard_included"] is True
    assert enclosure["below_declared"] is True
    assert enclosure["alpha_positive"] is True
    assert enclosure["dx_mag"] < 1.0 / 3.0
    honesty = payload["honesty"]
    assert honesty["orbit_continuation_on_kill"] is True
    assert honesty["g1_passed"] is False
    assert honesty["full_hilbert16_solved"] is False
    reasons = honesty["_honesty_reasons"]
    assert reasons["g1_passed"]["reason"] == "gate_open"
    assert "orbit_continuation_on_kill" not in reasons


def test_local_stage_b_never_earns_a_parent_flag() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_stage_b")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["full_hilbert16_solved"] is False
    assert flags["hilbert16_part_b_quadratic_solved"] is False
