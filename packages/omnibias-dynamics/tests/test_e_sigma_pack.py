# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""L in {9/25, 1/16} wall-span E_sigma cover; not Lohner-from-V=0 or G1."""

from __future__ import annotations

from fractions import Fraction

from omnibias.dynamics.e_sigma_pack import (
    KILL_L_PACK,
    identity_verdicts,
    report,
    residual_pack_align,
    residual_pack_hi,
    residual_pack_lo,
    residual_pack_width,
)
from omnibias.dynamics.hilbert16_ledger import default_h16_ledger, derived_parent_flags


def test_e_sigma_pack_identities_are_exact() -> None:
    h_lo, h_hi, align = Fraction(17, 1000), Fraction(7, 200), Fraction(1, 40)
    assert residual_pack_lo(h_lo) == 0
    assert residual_pack_hi(Fraction(35, 1000)) == 0
    assert residual_pack_width(h_lo, h_hi) == 0
    assert residual_pack_align(align, h_lo) == 0
    assert all(status == "PROVED" for status in identity_verdicts().values())
    assert KILL_L_PACK == (Fraction(9, 25), Fraction(1, 16))


def test_e_sigma_pack_hit_does_not_pass_g1() -> None:
    payload = report().to_payload()
    assert payload["schema"] == "hilbert16-e-sigma-pack-v1"
    assert payload["e_sigma_pack_hit"] is True
    assert payload["outgoing_first_hit"] is False
    assert set(payload["pack"]) == {"9/25", "1/16"}
    assert all(row["n_slabs"] == 18 and row["all_certified"] for row in payload["pack"].values())
    assert all(
        row["h_lo"] >= 0.017 and row["h_hi"] <= 0.035 for row in payload["walls"].values()
    )
    assert payload["coarse"]["all_certified"] is False
    assert payload["from_grazing_start"]["status"] != "certified"
    honesty = payload["honesty"]
    assert honesty["e_sigma_pack_hit"] is True
    assert honesty["outgoing_first_hit"] is False
    assert honesty["g1_passed"] is False
    assert honesty["full_hilbert16_solved"] is False
    reasons = honesty["_honesty_reasons"]
    assert reasons["g1_passed"]["reason"] == "gate_open"
    assert "e_sigma_pack_hit" not in reasons


def test_local_e_sigma_pack_never_earns_a_parent_flag() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_e_sigma_pack")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["full_hilbert16_solved"] is False
    assert flags["hilbert16_part_b_quadratic_solved"] is False
