# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Compact aligned parametric-eps E_sigma cover; not every eps, Z_x, or G1."""

from __future__ import annotations

from fractions import Fraction

from omnibias.dynamics.e_sigma_eps_span import (
    KILL_L_PACK,
    identity_verdicts,
    report,
    residual_espan_contains,
    residual_espan_lo,
    residual_espan_pack_L,
    residual_espan_width,
)
from omnibias.dynamics.hilbert16_ledger import default_h16_ledger, derived_parent_flags


def test_e_sigma_eps_span_identities_are_exact() -> None:
    eps_lo, eps_hi, mid = Fraction(1, 25), Fraction(1, 16), Fraction(1, 20)
    assert residual_espan_lo(eps_lo) == 0
    assert residual_espan_width(eps_lo, eps_hi) == 0
    assert residual_espan_contains(mid, eps_lo, eps_hi) == 0
    assert residual_espan_pack_L(Fraction(9, 25), Fraction(1, 16)) == 0
    assert all(status == "PROVED" for status in identity_verdicts().values())
    assert KILL_L_PACK == (Fraction(9, 25), Fraction(1, 16))


def test_e_sigma_eps_span_hit_does_not_pass_g1() -> None:
    payload = report().to_payload()
    assert payload["schema"] == "hilbert16-e-sigma-eps-span-v1"
    assert payload["e_sigma_eps_span_hit"] is True
    assert payload["outgoing_first_hit"] is False
    assert payload["cover"]["n_slabs"] == 3
    assert payload["cover"]["all_certified"] is True
    assert set(payload["pack"]) == {"9/25", "1/16"}
    assert all(
        row["status"] == "certified" and row["replayed"] and row["transverse_positive"]
        for row in payload["pack"].values()
    )
    assert payload["coarse"]["all_certified"] is False
    assert payload["from_grazing_start"]["status"] != "certified"
    honesty = payload["honesty"]
    assert honesty["e_sigma_eps_span_hit"] is True
    assert honesty["outgoing_first_hit"] is False
    assert honesty["g1_passed"] is False
    assert honesty["full_hilbert16_solved"] is False
    reasons = honesty["_honesty_reasons"]
    assert reasons["g1_passed"]["reason"] == "gate_open"
    assert "e_sigma_eps_span_hit" not in reasons


def test_local_e_sigma_eps_span_never_earns_a_parent_flag() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_e_sigma_eps_span")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["full_hilbert16_solved"] is False
    assert flags["hilbert16_part_b_quadratic_solved"] is False
