# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Kill-line matching-chart Lohner pack toward chart O; not complete first-hit or G1."""

from __future__ import annotations

from fractions import Fraction

from omnibias.dynamics.hilbert16_ledger import default_h16_ledger, derived_parent_flags
from omnibias.dynamics.stage_c_origin import (
    SEP_O_PACK,
    identity_verdicts,
    report,
    residual_r1_o_eighth,
    residual_r1_o_quarter,
    residual_r1_origin,
)


def test_stage_c_origin_identities_are_exact() -> None:
    assert residual_r1_origin() == 0
    assert residual_r1_o_quarter() == 0
    assert residual_r1_o_eighth() == 0
    assert Fraction(1) - Fraction(2) / 2 == 0
    assert Fraction(1) - Fraction(3, 2) / 2 == Fraction(1, 4)
    assert Fraction(1) - Fraction(7, 4) / 2 == Fraction(1, 8)
    assert all(status == "PROVED" for status in identity_verdicts().values())
    assert SEP_O_PACK == (Fraction(3, 2), Fraction(7, 4), Fraction(2))


def test_stage_c_origin_does_not_pass_g1() -> None:
    payload = report().to_payload()
    assert payload["schema"] == "hilbert16-stage-c-origin-v1"
    assert payload["stage_c_origin"] is True
    assert payload["outgoing_first_hit"] is False
    assert set(payload["pack"]) == {"3/2", "7/4", "2"}
    assert all(
        row["status"] == "certified"
        and row["replayed"]
        and row["transverse_positive"]
        and row["hits_section"]
        for row in payload["pack"].values()
    )
    assert payload["short"]["status"] != "certified"
    honesty = payload["honesty"]
    assert honesty["stage_c_origin"] is True
    assert honesty["outgoing_first_hit"] is False
    assert honesty["g1_passed"] is False
    assert honesty["full_hilbert16_solved"] is False
    reasons = honesty["_honesty_reasons"]
    assert reasons["g1_passed"]["reason"] == "gate_open"
    assert "stage_c_origin" not in reasons


def test_local_stage_c_origin_never_earns_a_parent_flag() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_stage_c_origin")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["full_hilbert16_solved"] is False
    assert flags["hilbert16_part_b_quadratic_solved"] is False
