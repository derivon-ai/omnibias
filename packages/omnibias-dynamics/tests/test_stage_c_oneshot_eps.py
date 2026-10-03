# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Finite shrinking-eps Stage-C Lohner pack; not every eps or G1."""

from __future__ import annotations

from fractions import Fraction

from omnibias.dynamics.hilbert16_ledger import default_h16_ledger, derived_parent_flags
from omnibias.dynamics.stage_c_oneshot_eps import (
    EPS_PACK,
    horizon_steps,
    identity_verdicts,
    report,
    residual_ceps_n20_sec,
    residual_ceps_n25_sec,
    residual_ceps_T25,
)


def test_stage_c_oneshot_eps_identities_are_exact() -> None:
    assert residual_ceps_n20_sec() == 0
    assert residual_ceps_n25_sec() == 0
    assert residual_ceps_T25() == 0
    assert Fraction(1, 4) / Fraction(1, 20) == Fraction(5)
    assert Fraction(1, 4) / Fraction(1, 25) == Fraction(25, 4)
    assert Fraction(160) * Fraction(1, 20) == Fraction(8)
    assert all(status == "PROVED" for status in identity_verdicts().values())
    assert EPS_PACK == (16, 20, 25)
    assert horizon_steps(16) == 120
    assert horizon_steps(20) == 140
    assert horizon_steps(25) == 160


def test_stage_c_oneshot_eps_does_not_pass_g1() -> None:
    payload = report().to_payload()
    assert payload["schema"] == "hilbert16-stage-c-oneshot-eps-v1"
    assert payload["stage_c_oneshot_eps"] is True
    assert payload["outgoing_first_hit"] is False
    assert set(payload["pack"]) == {"16", "20", "25"}
    assert all(
        row["status"] == "certified"
        and row["replayed"]
        and row["transverse_positive"]
        and row["hits_section"]
        for row in payload["pack"].values()
    )
    assert payload["short"]["status"] != "certified"
    honesty = payload["honesty"]
    assert honesty["stage_c_oneshot_eps"] is True
    assert honesty["stage_c_oneshot"] is False
    assert honesty["outgoing_first_hit"] is False
    assert honesty["g1_passed"] is False
    assert honesty["full_hilbert16_solved"] is False
    reasons = honesty["_honesty_reasons"]
    assert reasons["g1_passed"]["reason"] == "gate_open"
    assert "stage_c_oneshot_eps" not in reasons


def test_local_stage_c_oneshot_eps_never_earns_a_parent_flag() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_stage_c_oneshot_eps")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["full_hilbert16_solved"] is False
    assert flags["hilbert16_part_b_quadratic_solved"] is False
