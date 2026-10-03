# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Nearer-interface chart-O matching-chart Lohner cover from x=1/32; not complete first-hit or G1."""

from __future__ import annotations

from fractions import Fraction

from omnibias.dynamics.hilbert16_ledger import default_h16_ledger, derived_parent_flags
from omnibias.dynamics.stage_c_origin_x32 import (
    identity_verdicts,
    report,
    residual_ox32_join,
    residual_ox32_n128,
    residual_ox32_slabs,
)


def test_stage_c_origin_x32_identities_are_exact() -> None:
    assert residual_ox32_join() == 0
    assert residual_ox32_slabs() == 0
    assert residual_ox32_n128() == 0
    assert Fraction(31, 16) + Fraction(1, 16) == Fraction(2)
    assert Fraction(8) * Fraction(1, 128) == Fraction(1, 16)
    assert Fraction(128) * Fraction(1, 16) == Fraction(8)
    assert all(status == "PROVED" for status in identity_verdicts().values())


def test_stage_c_origin_x32_does_not_pass_g1() -> None:
    payload = report().to_payload()
    assert payload["schema"] == "hilbert16-stage-c-origin-x32-v1"
    assert payload["stage_c_origin_x32"] is True
    assert payload["outgoing_first_hit"] is False
    assert set(payload["pack"]) == {
        "31/16-249/128",
        "249/128-125/64",
        "125/64-251/128",
        "251/128-63/32",
        "63/32-253/128",
        "253/128-127/64",
        "127/64-255/128",
        "255/128-2",
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
    assert honesty["stage_c_origin_x32"] is True
    assert honesty["outgoing_first_hit"] is False
    assert honesty["g1_passed"] is False
    assert honesty["full_hilbert16_solved"] is False
    reasons = honesty["_honesty_reasons"]
    assert reasons["g1_passed"]["reason"] == "gate_open"
    assert "stage_c_origin_x32" not in reasons


def test_local_stage_c_origin_x32_never_earns_a_parent_flag() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_stage_c_origin_x32")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["full_hilbert16_solved"] is False
    assert flags["hilbert16_part_b_quadratic_solved"] is False
