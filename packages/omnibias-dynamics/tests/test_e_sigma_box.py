# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Wall-box h-interval E_sigma cover; not Lohner-from-V=0 or G1."""

from __future__ import annotations

from fractions import Fraction

from omnibias.dynamics.e_sigma_box import (
    KILL_L_PACK,
    identity_verdicts,
    report,
    residual_cover_contains,
    residual_cover_hi,
    residual_cover_lo,
    residual_cover_width,
)
from omnibias.dynamics.hilbert16_ledger import default_h16_ledger, derived_parent_flags


def test_e_sigma_box_identities_are_exact() -> None:
    h_lo, h_hi, align = Fraction(1, 50), Fraction(4, 125), Fraction(1, 40)
    assert residual_cover_lo(h_lo) == 0
    assert residual_cover_hi(h_hi) == 0
    assert residual_cover_width(h_lo, h_hi) == 0
    assert residual_cover_contains(align, h_lo, h_hi) == 0
    assert all(status == "PROVED" for status in identity_verdicts().values())
    assert KILL_L_PACK == (Fraction(9, 25), Fraction(1, 16), Fraction(0))


def test_e_sigma_box_hit_does_not_pass_g1() -> None:
    payload = report().to_payload()
    assert payload["schema"] == "hilbert16-e-sigma-box-v1"
    assert payload["e_sigma_box_hit"] is True
    assert payload["outgoing_first_hit"] is False
    assert payload["cover"]["n_slabs"] == 12
    assert payload["cover"]["all_certified"] is True
    assert set(payload["pack"]) == {"9/25", "1/16"}
    assert all(row["status"] == "certified" and row["replayed"] for row in payload["pack"].values())
    assert all(
        row["h_lo"] <= 0.02 and row["h_hi"] >= 0.032 for row in payload["walls"].values()
    )
    assert payload["coarse"]["all_certified"] is False
    assert payload["from_grazing_start"]["status"] != "certified"
    honesty = payload["honesty"]
    assert honesty["e_sigma_box_hit"] is True
    assert honesty["outgoing_first_hit"] is False
    assert honesty["g1_passed"] is False
    assert honesty["full_hilbert16_solved"] is False
    reasons = honesty["_honesty_reasons"]
    assert reasons["g1_passed"]["reason"] == "gate_open"
    assert "e_sigma_box_hit" not in reasons


def test_local_e_sigma_box_never_earns_a_parent_flag() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_e_sigma_box")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["full_hilbert16_solved"] is False
    assert flags["hilbert16_part_b_quadratic_solved"] is False
