# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Nearer-interface chart-O matching-chart Lohner cover; not complete first-hit or G1."""

from __future__ import annotations

from fractions import Fraction

from omnibias.dynamics.hilbert16_ledger import default_h16_ledger, derived_parent_flags
from omnibias.dynamics.stage_c_origin_iface import (
    identity_verdicts,
    report,
    residual_oiface_join,
    residual_oiface_n32,
    residual_oiface_slabs,
)


def test_stage_c_origin_iface_identities_are_exact() -> None:
    assert residual_oiface_join() == 0
    assert residual_oiface_slabs() == 0
    assert residual_oiface_n32() == 0
    assert Fraction(7, 4) + Fraction(1, 4) == Fraction(2)
    assert Fraction(8) * Fraction(1, 32) == Fraction(1, 4)
    assert Fraction(32) * Fraction(1, 4) == Fraction(8)
    assert all(status == "PROVED" for status in identity_verdicts().values())


def test_stage_c_origin_iface_does_not_pass_g1() -> None:
    payload = report().to_payload()
    assert payload["schema"] == "hilbert16-stage-c-origin-iface-v1"
    assert payload["stage_c_origin_iface"] is True
    assert payload["outgoing_first_hit"] is False
    assert set(payload["pack"]) == {
        "7/4-57/32",
        "57/32-29/16",
        "29/16-59/32",
        "59/32-15/8",
        "15/8-61/32",
        "61/32-31/16",
        "31/16-63/32",
        "63/32-2",
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
    assert honesty["stage_c_origin_iface"] is True
    assert honesty["outgoing_first_hit"] is False
    assert honesty["g1_passed"] is False
    assert honesty["full_hilbert16_solved"] is False
    reasons = honesty["_honesty_reasons"]
    assert reasons["g1_passed"]["reason"] == "gate_open"
    assert "stage_c_origin_iface" not in reasons


def test_local_stage_c_origin_iface_never_earns_a_parent_flag() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_stage_c_origin_iface")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["full_hilbert16_solved"] is False
    assert flags["hilbert16_part_b_quadratic_solved"] is False
