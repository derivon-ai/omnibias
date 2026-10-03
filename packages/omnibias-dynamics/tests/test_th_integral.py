# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Comparison-bootstrap T-h integral; not a Lohner orbit, first-hit, or G1."""

from __future__ import annotations

from fractions import Fraction

from omnibias.dynamics.hilbert16_ledger import default_h16_ledger, derived_parent_flags
from omnibias.dynamics.th_integral import (
    enclose_th_majorant,
    identity_verdicts,
    report,
    residual_ftc_linear,
    residual_log_sqrt_prefactor,
    residual_majorant_split,
    residual_sqrt_ratio_sq,
    residual_th_dh,
)


def test_th_integral_identities_are_exact() -> None:
    q, h, kay = Fraction(3, 16), Fraction(1, 4), Fraction(17, 16)
    c, kbound, eps = Fraction(2), Fraction(2), Fraction(1, 16)
    te, he, alpha = Fraction(1, 8), Fraction(1, 16), Fraction(1, 4)
    s, y0, hmax = Fraction(32), Fraction(4), Fraction(1)
    assert residual_th_dh(q, h, kay) == 0
    assert residual_majorant_split(c, kbound, eps, h) == 0
    assert residual_ftc_linear(te, he, alpha, h) == 0
    assert residual_sqrt_ratio_sq(eps, s, y0, hmax) == 0
    assert residual_log_sqrt_prefactor(eps, s, y0, hmax) == 0
    assert all(status == "PROVED" for status in identity_verdicts().values())


def test_th_integral_majorant_does_not_pass_g1() -> None:
    payload = report().to_payload()
    assert payload["schema"] == "hilbert16-th-integral-v1"
    assert payload["th_integral_majorant"] is True
    assert payload["outgoing_first_hit"] is False
    assert payload["majorant"]["within_margin"] is True
    assert payload["majorant"]["te_above_he"] is True
    honesty = payload["honesty"]
    assert honesty["th_integral_majorant"] is True
    assert honesty["outgoing_first_hit"] is False
    assert honesty["g1_passed"] is False
    assert honesty["full_hilbert16_solved"] is False
    reasons = honesty["_honesty_reasons"]
    assert reasons["g1_passed"]["reason"] == "gate_open"
    assert "th_integral_majorant" not in reasons


def test_th_integral_refuses_when_exit_gap_vanishes() -> None:
    wide = enclose_th_majorant(ns=(8,))
    assert wide["te_above_he"] is False
    assert wide["within_margin"] is False


def test_local_th_integral_never_earns_a_parent_flag() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_th_integral")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["full_hilbert16_solved"] is False
    assert flags["hilbert16_part_b_quadratic_solved"] is False
