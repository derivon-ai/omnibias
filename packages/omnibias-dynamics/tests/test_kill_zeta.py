# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Kill-compact Cauchy majorant on lambda1=-2 including L=0; not G1."""

from __future__ import annotations

from fractions import Fraction

from omnibias.dynamics.hilbert16_ledger import default_h16_ledger, derived_parent_flags
from omnibias.dynamics.kill_zeta import (
    cauchy_majorant_kill,
    identity_verdicts,
    kill_L,
    report,
    residual_kill_disc_gap,
    residual_kill_lambda,
    residual_kill_product,
    residual_kill_sum,
    residual_kill_tworoot,
    sample_z_kill,
)


def test_kill_zeta_identities_are_exact() -> None:
    r1, lam1 = Fraction(1, 5), Fraction(-2)
    r2 = 2 - r1
    L = kill_L(r1)
    assert residual_kill_lambda(lam1) == 0
    assert residual_kill_product(L, r1) == 0
    assert residual_kill_sum(r1, r2) == 0
    assert residual_kill_disc_gap(lam1, L) == 0
    assert residual_kill_tworoot(Fraction(1), r1, r2, L) == 0
    assert 4 * (1 - L) > 0
    assert all(status == "PROVED" for status in identity_verdicts().values())


def test_kill_cauchy_includes_L_zero_and_does_not_pass_g1() -> None:
    payload = report().to_payload()
    assert payload["schema"] == "hilbert16-kill-zeta-v1"
    sample = sample_z_kill()
    assert payload["sample_abs_z"] == sample.abs().hi
    assert payload["cauchy"]["finite"] is True
    assert payload["cauchy"]["picard_included"] is True
    assert payload["cauchy"]["L_lo"] == 0.0
    assert payload["cauchy"]["L_hi"] == 1.0
    assert payload["sample_below_bound"] is True
    honesty = payload["honesty"]
    assert honesty["kill_z_compact_bound"] is True
    assert honesty["fold_z_compact_bound"] is False
    assert honesty["outgoing_first_hit"] is False
    assert honesty["g1_passed"] is False
    assert honesty["full_hilbert16_solved"] is False
    reasons = honesty["_honesty_reasons"]
    assert reasons["g1_passed"]["reason"] == "gate_open"
    assert reasons["outgoing_first_hit"]["reason"] == "unimplemented"
    assert "kill_z_compact_bound" not in reasons


def test_larger_kill_nu_box_refuses_picard() -> None:
    wide = cauchy_majorant_kill(radius_nu=0.05, k_rad=0.4)
    assert wide["picard_included"] is False
    assert wide["finite"] is False


def test_local_kill_zeta_never_earns_a_parent_flag() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_kill_zeta")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["full_hilbert16_solved"] is False
    assert flags["hilbert16_part_b_quadratic_solved"] is False
