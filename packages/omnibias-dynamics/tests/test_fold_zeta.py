# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Fold-compact Cauchy majorant; not physical C2 or G1."""

from __future__ import annotations

from fractions import Fraction

from omnibias.dynamics.fold_zeta import (
    cauchy_majorant_fold,
    fold_lambda,
    identity_verdicts,
    report,
    residual_fold_bminus,
    residual_fold_disc,
    residual_fold_L,
    residual_fold_lambda,
    sample_z_fold,
)
from omnibias.dynamics.hilbert16_ledger import default_h16_ledger, derived_parent_flags


def test_fold_disc_identities_are_exact() -> None:
    rstar, x = Fraction(3, 2), Fraction(2)
    lam0, lam1 = fold_lambda(rstar)
    assert residual_fold_lambda(rstar, lam1) == 0
    assert residual_fold_L(rstar, lam0) == 0
    assert residual_fold_disc(lam0, lam1) == 0
    assert residual_fold_bminus(x, rstar) == 0
    assert all(status == "PROVED" for status in identity_verdicts().values())


def test_fold_cauchy_dominates_sample_and_does_not_pass_g1() -> None:
    payload = report().to_payload()
    assert payload["schema"] == "hilbert16-fold-zeta-v1"
    sample = sample_z_fold()
    assert payload["sample_abs_z"] == sample.abs().hi
    assert payload["cauchy"]["finite"] is True
    assert payload["cauchy"]["picard_included"] is True
    assert payload["sample_below_bound"] is True
    honesty = payload["honesty"]
    assert honesty["fold_z_compact_bound"] is True
    assert honesty["inner_z_compact_bound"] is False
    assert honesty["g1_passed"] is False
    assert honesty["g4_passed"] is False
    assert honesty["full_hilbert16_solved"] is False
    assert honesty["fold_leading_c2_remainder"] is False
    reasons = honesty["_honesty_reasons"]
    assert reasons["g1_passed"]["reason"] == "gate_open"
    assert reasons["fold_leading_c2_remainder"]["reason"] == "structural_limit"
    assert "fold_z_compact_bound" not in reasons


def test_larger_fold_box_refuses_picard() -> None:
    wide = cauchy_majorant_fold(
        radius_v=0.08,
        radius_nu=0.03125,
        rstar_lo=1.25,
        rstar_hi=1.75,
        k_rad=0.5,
    )
    assert wide["picard_included"] is False
    assert wide["finite"] is False


def test_local_fold_zeta_never_earns_a_parent_flag() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_fold_zeta")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["full_hilbert16_solved"] is False
    assert flags["hilbert16_part_b_quadratic_solved"] is False
