# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""L=0 whole-wall h-span E_sigma cover; not Lohner-from-V=0 or G1."""

from __future__ import annotations

from fractions import Fraction

from omnibias.dynamics.e_sigma_span import (
    identity_verdicts,
    report,
    residual_span_align_hi,
    residual_span_align_lo,
    residual_span_hi,
    residual_span_width,
)
from omnibias.dynamics.hilbert16_ledger import default_h16_ledger, derived_parent_flags


def test_e_sigma_span_identities_are_exact() -> None:
    h_lo, h_hi, align = Fraction(19, 1000), Fraction(1, 25), Fraction(1, 40)
    assert residual_span_hi(Fraction(40, 1000)) == 0
    assert residual_span_width(h_lo, h_hi) == 0
    assert residual_span_align_lo(align, h_lo) == 0
    assert residual_span_align_hi(h_hi, align) == 0
    assert all(status == "PROVED" for status in identity_verdicts().values())


def test_e_sigma_span_hit_does_not_pass_g1() -> None:
    payload = report().to_payload()
    assert payload["schema"] == "hilbert16-e-sigma-span-v1"
    assert payload["e_sigma_span_hit"] is True
    assert payload["outgoing_first_hit"] is False
    assert payload["cover"]["n_slabs"] == 21
    assert payload["cover"]["all_certified"] is True
    assert payload["wall"]["h_lo"] >= 0.019
    assert payload["wall"]["h_hi"] <= 0.04
    assert payload["coarse"]["all_certified"] is False
    assert payload["from_grazing_start"]["status"] != "certified"
    honesty = payload["honesty"]
    assert honesty["e_sigma_span_hit"] is True
    assert honesty["outgoing_first_hit"] is False
    assert honesty["g1_passed"] is False
    assert honesty["full_hilbert16_solved"] is False
    reasons = honesty["_honesty_reasons"]
    assert reasons["g1_passed"]["reason"] == "gate_open"
    assert "e_sigma_span_hit" not in reasons


def test_local_e_sigma_span_never_earns_a_parent_flag() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_e_sigma_span")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["full_hilbert16_solved"] is False
    assert flags["hilbert16_part_b_quadratic_solved"] is False
