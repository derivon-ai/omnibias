# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Kill-line sep*S_pre on every sep in (0, 1]; not off-kill dx_e or G1."""

from __future__ import annotations

from fractions import Fraction

from omnibias.dynamics.hilbert16_ledger import default_h16_ledger, derived_parent_flags
from omnibias.dynamics.sep_spre import (
    identity_verdicts,
    report,
    residual_spre_chi,
    residual_spre_net,
    residual_spre_tail,
)


def test_sep_spre_identities_are_exact() -> None:
    assert residual_spre_tail() == 0
    assert residual_spre_chi() == 0
    assert residual_spre_net() == 0
    assert Fraction(2**48) * Fraction(1, 2**48) == 1
    assert Fraction(4) / Fraction(1, 2) == 8
    assert Fraction(3, 16) * (4 - Fraction(11, 5)) == Fraction(27, 80)
    assert Fraction(11, 5) < 3
    assert all(status == "PROVED" for status in identity_verdicts().values())


def test_sep_spre_does_not_pass_g1() -> None:
    payload = report().to_payload()
    assert payload["schema"] == "hilbert16-sep-spre-v1"
    assert payload["sep_spre"] is True
    assert payload["outgoing_first_hit"] is False
    enclosure = payload["enclosure"]
    assert enclosure["below_cap"] is True
    assert enclosure["slabs"] == 48
    assert float(enclosure["s_hi"]) < float(Fraction(11, 5))
    assert float(enclosure["after_lo"]) > 0.125
    assert float(enclosure["net_lo"]) > float(Fraction(27, 80))
    assert enclosure["chi_below_declared"] is True
    assert enclosure["decreasing"] is True
    stall = payload["stall"]
    assert stall["finite"] is False
    assert stall["net_below_declared"] is False
    honesty = payload["honesty"]
    assert honesty["sep_spre"] is True
    assert honesty["dx_e_leading"] is False
    assert honesty["outgoing_first_hit"] is False
    assert honesty["g1_passed"] is False
    assert honesty["full_hilbert16_solved"] is False
    reasons = honesty["_honesty_reasons"]
    assert reasons["g1_passed"]["reason"] == "gate_open"
    assert "sep_spre" not in reasons


def test_local_sep_spre_never_earns_a_parent_flag() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_sep_spre")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["full_hilbert16_solved"] is False
    assert flags["hilbert16_part_b_quadratic_solved"] is False
