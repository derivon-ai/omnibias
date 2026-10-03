# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Exact fold I-map derivatives; not a physical remainder or G1 certificate."""

from __future__ import annotations

from fractions import Fraction

from omnibias.dynamics.fold_leading import (
    fold_dx_dkappa,
    fold_log_d2_dkappa2,
    fold_reciprocal_dx,
    fold_reciprocal_log_c2,
    identity_verdicts,
    report,
    residual_chi_log_second,
    residual_fold_d2x,
    residual_fold_dx,
    residual_fold_log_d1,
    residual_fold_log_d2,
    residual_fold_reciprocal,
    residual_fold_reciprocal_log_c2,
    residual_interface_relative,
    residual_lifted_d2x,
    residual_lifted_dx,
    residual_lifted_log_c2,
    residual_lifted_log_d1,
    residual_relative_remainder,
    residual_z_relative_on_lift,
    residual_zeta_rho,
)
from omnibias.dynamics.hilbert16_ledger import default_h16_ledger, derived_parent_flags


def test_fold_imap_identities_are_exact() -> None:
    delta, rstar, kappa = Fraction(1), Fraction(3), Fraction(3)
    assert residual_fold_dx(delta, rstar) == 0
    assert residual_fold_d2x(delta, rstar) == 0
    assert residual_fold_log_d1(delta, rstar) == 0
    assert residual_fold_log_d2(delta, rstar) == 0
    assert residual_fold_reciprocal(rstar, kappa) == 0
    assert residual_fold_reciprocal_log_c2(rstar, kappa) == 0
    assert residual_relative_remainder(
        Fraction(2), Fraction(1, 4), Fraction(3), Fraction(19, 4)
    ) == 0
    assert residual_interface_relative(Fraction(2), Fraction(1), Fraction(3), Fraction(4)) == 0
    assert residual_zeta_rho(
        Fraction(2), Fraction(9, 4), Fraction(-3), Fraction(1, 5), Fraction(1, 3), Fraction(7, 2)
    ) == 0
    assert residual_lifted_dx(Fraction(2), Fraction(3), Fraction(1, 4)) == 0
    assert residual_lifted_d2x(Fraction(2), Fraction(3), Fraction(1, 4)) == 0
    assert residual_lifted_log_d1(Fraction(2), Fraction(3), Fraction(1, 4)) == 0
    assert residual_lifted_log_c2(Fraction(2), Fraction(3), Fraction(1, 4)) == 0
    assert residual_z_relative_on_lift(Fraction(1, 5), Fraction(1, 3), Fraction(7, 2)) == 0
    assert residual_chi_log_second(
        Fraction(2), Fraction(1, 5), Fraction(3, 2), Fraction(4), Fraction(1, 3)
    ) == 0
    assert fold_dx_dkappa(delta, rstar) == Fraction(1, 2)
    assert fold_reciprocal_dx(rstar, kappa) == Fraction(1, 2)
    assert fold_log_d2_dkappa2(delta, rstar) == fold_reciprocal_log_c2(kappa)
    assert all(status == "PROVED" for status in identity_verdicts().values())


def test_large_kappa_sensitivity_decays_and_does_not_pass_g1() -> None:
    payload = report(rstar=1.5, kappa=20.0).to_payload()
    assert payload["dx_decays"] is True
    assert payload["chi_c2_vanishes"] is True
    assert payload["inner_c2_o_eps"] is True
    assert payload["large_kappa_dx"] < 1.5 / (20.0 * 19.0) + 1e-12
    honesty = payload["honesty"]
    assert honesty["g1_passed"] is False
    assert honesty["g4_passed"] is False
    assert honesty["full_hilbert16_solved"] is False
    assert honesty["fold_leading_c2_remainder"] is False
    assert honesty["inner_z_compact_bound"] is False
    reasons = honesty["_honesty_reasons"]
    assert reasons["g1_passed"]["reason"] == "gate_open"
    assert reasons["fold_leading_c2_remainder"]["reason"] == "structural_limit"
    assert reasons["inner_z_compact_bound"]["reason"] == "unimplemented"


def test_local_fold_imap_never_earns_a_parent_flag() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_fold_imap")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["full_hilbert16_solved"] is False
    assert flags["hilbert16_part_b_quadratic_solved"] is False
