# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Frozen-Z C2 identities; not a physical remainder or G1 certificate."""

from __future__ import annotations

from fractions import Fraction

from omnibias.dynamics.hilbert16_ledger import default_h16_ledger, derived_parent_flags
from omnibias.dynamics.physical_c2 import (
    frozen_c2_gap,
    frozen_log_d1_gap,
    identity_verdicts,
    report,
    residual_frozen_c2_gap,
    residual_frozen_log_d1_gap,
    residual_lift_cubic_mismatch,
)


def test_frozen_c2_identities_are_exact() -> None:
    x, rstar, mu, eps, z_jet, beta0 = (
        Fraction(2),
        Fraction(3, 2),
        Fraction(1, 4),
        Fraction(1, 5),
        Fraction(7, 2),
        Fraction(1, 3),
    )
    assert residual_lift_cubic_mismatch(x, rstar, beta0, eps) == 0
    assert residual_frozen_log_d1_gap(x, rstar, mu, eps, z_jet) == 0
    assert residual_frozen_c2_gap(x, rstar, mu, eps, z_jet) == 0
    assert frozen_log_d1_gap(x, rstar, mu, eps, z_jet) == 2 * eps**2 * x**3 * z_jet
    assert frozen_c2_gap(x, rstar, mu, eps, z_jet) * x == 2 * eps**2 * z_jet * (
        (x - rstar) ** 2 + mu + eps**2 * x**3 * z_jet
    )
    assert all(status == "PROVED" for status in identity_verdicts().values())


def test_physical_c2_does_not_pass_g1() -> None:
    payload = report().to_payload()
    assert payload["schema"] == "hilbert16-physical-c2-v1"
    honesty = payload["honesty"]
    assert honesty["physical_c2_remainder"] is False
    assert honesty["g1_passed"] is False
    assert honesty["g4_passed"] is False
    assert honesty["full_hilbert16_solved"] is False
    reasons = honesty["_honesty_reasons"]
    assert reasons["g1_passed"]["reason"] == "gate_open"
    assert reasons["physical_c2_remainder"]["reason"] == "unimplemented"


def test_local_physical_c2_never_earns_a_parent_flag() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_physical_c2")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["full_hilbert16_solved"] is False
    assert flags["hilbert16_part_b_quadratic_solved"] is False
