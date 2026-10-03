# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Algebraic r=-1 zeta identities; Cauchy majorant is not G1."""

from __future__ import annotations

from fractions import Fraction

from omnibias.dynamics.canonical_zeta import (
    SAMPLE_K,
    SAMPLE_LAM0,
    SAMPLE_LAM1,
    SAMPLE_NU,
    SAMPLE_V0,
    identity_verdicts,
    k_lambda0,
    report,
    residual_beta_linear_jet,
    residual_c0_jet,
    residual_field_matches_closed,
    residual_k_implicit,
    residual_k_lambda0,
    residual_limiting_cubic,
    residual_sqrt_inverse,
    residual_v0_quadratic,
    residual_V_factor,
    residual_zeta_at_zero,
    residual_zeta_plus_one_canceled,
    sample_z,
    slow_line_V,
    zeta_closed,
)
from omnibias.dynamics.hilbert16_ledger import default_h16_ledger, derived_parent_flags


def test_canonical_zeta_identities_are_exact() -> None:
    nu, v0, v = SAMPLE_NU, SAMPLE_V0, Fraction(1, 2)
    v_coord = slow_line_V(nu, v)
    assert residual_v0_quadratic(nu, v0) == 0
    assert residual_sqrt_inverse(nu, v, v_coord) == 0
    assert residual_V_factor(nu, v, v0, v_coord) == 0
    assert residual_zeta_at_zero(nu, v0) == 0
    assert residual_limiting_cubic(Fraction(2)) == 0
    assert residual_k_lambda0(nu, v0) == 0
    assert residual_beta_linear_jet(nu, v0) == 0
    assert residual_zeta_plus_one_canceled(nu, v, v0) == 0
    assert residual_k_implicit(nu, v0, SAMPLE_K, SAMPLE_LAM0, SAMPLE_LAM1) == 0
    assert residual_field_matches_closed(nu, v, v0) == 0
    assert residual_c0_jet(nu, v0, SAMPLE_K, Fraction(1, 7), SAMPLE_LAM1) == 0
    assert zeta_closed(nu, v0, v0) == -1
    assert k_lambda0(nu, v0) == Fraction(8, 5)
    assert all(status == "PROVED" for status in identity_verdicts().values())


def test_cauchy_majorant_dominates_sample_and_does_not_pass_g1() -> None:
    payload = report().to_payload()
    assert payload["lambda_slice"] == "lambda0=lambda1=0"
    assert payload["sample_abs_z"] == abs(float(sample_z()))
    assert payload["cauchy"]["finite"] is True
    assert payload["sample_below_bound"] is True
    honesty = payload["honesty"]
    assert honesty["inner_z_compact_bound"] is True
    assert honesty["fold_z_compact_bound"] is False
    assert honesty["g1_passed"] is False
    assert honesty["g4_passed"] is False
    assert honesty["full_hilbert16_solved"] is False
    assert honesty["fold_leading_c2_remainder"] is False
    reasons = honesty["_honesty_reasons"]
    assert reasons["g1_passed"]["reason"] == "gate_open"
    assert reasons["fold_leading_c2_remainder"]["reason"] == "structural_limit"


def test_local_canonical_zeta_never_earns_a_parent_flag() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_canonical_zeta")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["full_hilbert16_solved"] is False
    assert flags["hilbert16_part_b_quadratic_solved"] is False
