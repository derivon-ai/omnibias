# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Nearer-interface chart-O matching-chart Lohner cover from x=1/16; not complete first-hit or G1."""

from __future__ import annotations

from fractions import Fraction

from omnibias.dynamics.hilbert16_ledger import default_h16_ledger, derived_parent_flags
from omnibias.dynamics.stage_c_origin_near import (
    identity_verdicts,
    report,
    residual_onear_join,
    residual_onear_n64,
    residual_onear_slabs,
)


def test_stage_c_origin_near_identities_are_exact() -> None:
    assert residual_onear_join() == 0
    assert residual_onear_slabs() == 0
    assert residual_onear_n64() == 0
    assert Fraction(15, 8) + Fraction(1, 8) == Fraction(2)
    assert Fraction(8) * Fraction(1, 64) == Fraction(1, 8)
    assert Fraction(64) * Fraction(1, 8) == Fraction(8)
    assert all(status == "PROVED" for status in identity_verdicts().values())


def test_stage_c_origin_near_does_not_pass_g1() -> None:
    payload = report().to_payload()
    assert payload["schema"] == "hilbert16-stage-c-origin-near-v1"
    assert payload["stage_c_origin_near"] is True
    assert payload["outgoing_first_hit"] is False
    assert set(payload["pack"]) == {
        "15/8-121/64",
        "121/64-61/32",
        "61/32-123/64",
        "123/64-31/16",
        "31/16-125/64",
        "125/64-63/32",
        "63/32-127/64",
        "127/64-2",
    }
    assert all(
        row["status"] == "certified"
        and row["replayed"]
        and row["transverse_positive"]
        and row["hits_section"]
        for row in payload["pack"].values()
    )
    assert payload["coarse"]["status"] != "certified"
    assert payload["short"]["status"] != "certified"
    honesty = payload["honesty"]
    assert honesty["stage_c_origin_near"] is True
    assert honesty["outgoing_first_hit"] is False
    assert honesty["g1_passed"] is False
    assert honesty["full_hilbert16_solved"] is False
    reasons = honesty["_honesty_reasons"]
    assert reasons["g1_passed"]["reason"] == "gate_open"
    assert "stage_c_origin_near" not in reasons


def test_local_stage_c_origin_near_never_earns_a_parent_flag() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_stage_c_origin_near")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["full_hilbert16_solved"] is False
    assert flags["hilbert16_part_b_quadratic_solved"] is False
