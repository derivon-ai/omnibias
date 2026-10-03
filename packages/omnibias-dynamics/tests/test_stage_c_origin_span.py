# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Parametric-sep chart-O matching-chart Lohner cover; not complete first-hit or G1."""

from __future__ import annotations

from fractions import Fraction

from omnibias.dynamics.hilbert16_ledger import default_h16_ledger, derived_parent_flags
from omnibias.dynamics.stage_c_origin_span import (
    identity_verdicts,
    report,
    residual_ospan_join,
    residual_ospan_n16,
    residual_ospan_slabs,
)


def test_stage_c_origin_span_identities_are_exact() -> None:
    assert residual_ospan_join() == 0
    assert residual_ospan_slabs() == 0
    assert residual_ospan_n16() == 0
    assert Fraction(3, 2) + Fraction(1, 2) == Fraction(2)
    assert Fraction(8) * Fraction(1, 16) == Fraction(1, 2)
    assert Fraction(16) * Fraction(1, 2) == Fraction(8)
    assert all(status == "PROVED" for status in identity_verdicts().values())


def test_stage_c_origin_span_does_not_pass_g1() -> None:
    payload = report().to_payload()
    assert payload["schema"] == "hilbert16-stage-c-origin-span-v1"
    assert payload["stage_c_origin_span"] is True
    assert payload["outgoing_first_hit"] is False
    assert set(payload["pack"]) == {
        "3/2-25/16",
        "25/16-13/8",
        "13/8-27/16",
        "27/16-7/4",
        "7/4-29/16",
        "29/16-15/8",
        "15/8-31/16",
        "31/16-2",
    }
    assert all(
        row["status"] == "certified"
        and row["replayed"]
        and row["transverse_positive"]
        and row["hits_section"]
        for row in payload["pack"].values()
    )
    assert payload["coarse"]["status"] != "certified"
    honesty = payload["honesty"]
    assert honesty["stage_c_origin_span"] is True
    assert honesty["outgoing_first_hit"] is False
    assert honesty["g1_passed"] is False
    assert honesty["full_hilbert16_solved"] is False
    reasons = honesty["_honesty_reasons"]
    assert reasons["g1_passed"]["reason"] == "gate_open"
    assert "stage_c_origin_span" not in reasons


def test_local_stage_c_origin_span_never_earns_a_parent_flag() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_stage_c_origin_span")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["full_hilbert16_solved"] is False
    assert flags["hilbert16_part_b_quadratic_solved"] is False
