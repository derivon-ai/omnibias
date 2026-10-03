# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Unfrozen-Z first-log-derivative identities; not a Z_x bound or G1."""

from __future__ import annotations

from fractions import Fraction

from omnibias.dynamics.hilbert16_ledger import default_h16_ledger, derived_parent_flags
from omnibias.dynamics.physical_c2 import frozen_log_d1_gap
from omnibias.dynamics.z_x_gap import (
    identity_verdicts,
    report,
    residual_unfrozen_log_d1_gap,
    residual_unfrozen_recovers_frozen,
    residual_zx_extra,
    unfrozen_log_d1_gap,
)


def test_unfrozen_zx_identities_are_exact() -> None:
    x, rstar, mu, eps, z_jet, z_x = (
        Fraction(2),
        Fraction(3, 2),
        Fraction(1, 4),
        Fraction(1, 5),
        Fraction(7, 2),
        Fraction(5, 2),
    )
    assert residual_unfrozen_log_d1_gap(x, rstar, mu, eps, z_jet, z_x) == 0
    assert residual_unfrozen_recovers_frozen(x, rstar, mu, eps, z_jet) == 0
    assert residual_zx_extra(x, rstar, mu, eps, z_jet, z_x) == 0
    assert unfrozen_log_d1_gap(x, rstar, mu, eps, z_jet, z_x) == eps**2 * (
        2 * x**3 * z_jet + x**4 * z_x
    )
    assert unfrozen_log_d1_gap(x, rstar, mu, eps, z_jet, Fraction(0)) == frozen_log_d1_gap(
        x, rstar, mu, eps, z_jet
    )
    assert all(status == "PROVED" for status in identity_verdicts().values())


def test_z_x_gap_does_not_pass_g1() -> None:
    payload = report().to_payload()
    assert payload["schema"] == "hilbert16-z-x-gap-v1"
    honesty = payload["honesty"]
    assert honesty["z_x_bound"] is False
    assert honesty["physical_c2_remainder"] is False
    assert honesty["g1_passed"] is False
    assert honesty["g4_passed"] is False
    assert honesty["full_hilbert16_solved"] is False
    reasons = honesty["_honesty_reasons"]
    assert reasons["g1_passed"]["reason"] == "gate_open"
    assert reasons["z_x_bound"]["reason"] == "unimplemented"


def test_local_z_x_gap_never_earns_a_parent_flag() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_z_x_gap")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["full_hilbert16_solved"] is False
    assert flags["hilbert16_part_b_quadratic_solved"] is False
