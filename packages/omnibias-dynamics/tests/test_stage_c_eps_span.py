# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Compact parametric-eps Stage-C Lohner cover; not every eps or G1."""

from __future__ import annotations

from fractions import Fraction

from omnibias.dynamics.hilbert16_ledger import default_h16_ledger, derived_parent_flags
from omnibias.dynamics.stage_c_eps_span import (
    identity_verdicts,
    report,
    residual_cspan_join,
    residual_cspan_n800,
    residual_cspan_slabs,
)


def test_stage_c_eps_span_identities_are_exact() -> None:
    assert residual_cspan_join() == 0
    assert residual_cspan_slabs() == 0
    assert residual_cspan_n800() == 0
    assert Fraction(23, 400) + Fraction(1, 200) == Fraction(1, 16)
    assert Fraction(4) * Fraction(1, 800) == Fraction(1, 200)
    assert Fraction(800) * Fraction(1, 16) == Fraction(50)
    assert all(status == "PROVED" for status in identity_verdicts().values())


def test_stage_c_eps_span_does_not_pass_g1() -> None:
    payload = report().to_payload()
    assert payload["schema"] == "hilbert16-stage-c-eps-span-v1"
    assert payload["stage_c_eps_span"] is True
    assert payload["outgoing_first_hit"] is False
    assert set(payload["pack"]) == {
        "23/400-47/800",
        "47/800-3/50",
        "3/50-49/800",
        "49/800-1/16",
    }
    assert all(
        row["status"] == "certified" and row["replayed"] and row["transverse_positive"]
        for row in payload["pack"].values()
    )
    assert payload["coarse"]["status"] != "certified"
    honesty = payload["honesty"]
    assert honesty["stage_c_eps_span"] is True
    assert honesty["stage_c_oneshot_eps"] is False
    assert honesty["outgoing_first_hit"] is False
    assert honesty["g1_passed"] is False
    assert honesty["full_hilbert16_solved"] is False
    reasons = honesty["_honesty_reasons"]
    assert reasons["g1_passed"]["reason"] == "gate_open"
    assert "stage_c_eps_span" not in reasons


def test_local_stage_c_eps_span_never_earns_a_parent_flag() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_stage_c_eps_span")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["full_hilbert16_solved"] is False
    assert flags["hilbert16_part_b_quadratic_solved"] is False
