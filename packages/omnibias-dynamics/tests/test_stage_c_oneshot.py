# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Kill-line Stage-C Lohner first-hit of matching-chart x=4; not G1."""

from __future__ import annotations

from fractions import Fraction

from omnibias.dynamics.hilbert16_ledger import default_h16_ledger, derived_parent_flags
from omnibias.dynamics.stage_c_oneshot import (
    SEP_PACK,
    identity_verdicts,
    report,
    residual_hit_T,
    residual_short_T,
    residual_x_section,
)


def test_stage_c_oneshot_identities_are_exact() -> None:
    assert residual_x_section() == 0
    assert residual_hit_T() == 0
    assert residual_short_T() == 0
    assert Fraction(1, 4) / Fraction(1, 16) == Fraction(4)
    assert Fraction(120) * Fraction(1, 20) == Fraction(6)
    assert Fraction(80) * Fraction(1, 20) == Fraction(4)
    assert all(status == "PROVED" for status in identity_verdicts().values())
    assert SEP_PACK == (Fraction(0), Fraction(3, 5), Fraction(1))


def test_stage_c_oneshot_does_not_pass_g1() -> None:
    payload = report().to_payload()
    assert payload["schema"] == "hilbert16-stage-c-oneshot-v1"
    assert payload["stage_c_oneshot"] is True
    assert payload["outgoing_first_hit"] is False
    assert set(payload["pack"]) == {"0", "3/5", "1"}
    assert all(
        row["status"] == "certified"
        and row["replayed"]
        and row["transverse_positive"]
        and row["hits_section"]
        for row in payload["pack"].values()
    )
    assert payload["short"]["status"] != "certified"
    honesty = payload["honesty"]
    assert honesty["stage_c_oneshot"] is True
    assert honesty["stage_c_sec"] is False
    assert honesty["outgoing_first_hit"] is False
    assert honesty["g1_passed"] is False
    assert honesty["full_hilbert16_solved"] is False
    reasons = honesty["_honesty_reasons"]
    assert reasons["g1_passed"]["reason"] == "gate_open"
    assert "stage_c_oneshot" not in reasons


def test_local_stage_c_oneshot_never_earns_a_parent_flag() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_stage_c_oneshot")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["full_hilbert16_solved"] is False
    assert flags["hilbert16_part_b_quadratic_solved"] is False
