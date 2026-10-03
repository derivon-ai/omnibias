# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Kill-line Stage-C a_min; not an outgoing orbit, first-hit, or G1."""

from __future__ import annotations

from fractions import Fraction

from omnibias.dynamics.hilbert16_ledger import default_h16_ledger, derived_parent_flags
from omnibias.dynamics.stage_b import r1_kill
from omnibias.dynamics.stage_c import (
    identity_verdicts,
    report,
    residual_amin_floor,
    residual_amin_written,
    residual_inv_guess,
)


def test_stage_c_identities_are_exact() -> None:
    assert residual_amin_written() == 0
    assert residual_inv_guess() == 0
    assert residual_amin_floor() == 0
    assert r1_kill(Fraction(3, 5)) == Fraction(7, 10)
    assert 1 / r1_kill(Fraction(3, 5)) == Fraction(10, 7)
    assert all(status == "PROVED" for status in identity_verdicts().values())


def test_stage_c_does_not_pass_g1() -> None:
    payload = report().to_payload()
    assert payload["schema"] == "hilbert16-stage-c-v1"
    assert payload["stage_c_amin"] is True
    assert payload["sample_inside"] is True
    enclosure = payload["enclosure"]
    assert enclosure["below_declared"] is True
    assert enclosure["amin_above_floor"] is True
    assert enclosure["excludes_zero"] is True
    assert enclosure["picard_included"] is True
    assert enclosure["end_lo"] > 0.25
    assert enclosure["inv_hi"] < 8.0
    honesty = payload["honesty"]
    assert honesty["stage_c_amin"] is True
    assert honesty["dx_e_unif"] is False
    assert honesty["g1_passed"] is False
    assert honesty["full_hilbert16_solved"] is False
    reasons = honesty["_honesty_reasons"]
    assert reasons["g1_passed"]["reason"] == "gate_open"
    assert "stage_c_amin" not in reasons


def test_local_stage_c_never_earns_a_parent_flag() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_stage_c")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["full_hilbert16_solved"] is False
    assert flags["hilbert16_part_b_quadratic_solved"] is False
