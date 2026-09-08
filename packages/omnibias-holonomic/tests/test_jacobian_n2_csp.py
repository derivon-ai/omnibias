# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Inverse Keller CSP / soft-opt track for Jacobian n=2."""

from __future__ import annotations

from fractions import Fraction

from omnibias.holonomic.jacobian_n2 import shear_map
from omnibias.holonomic.jacobian_n2_csp import (
    anneal_keller_propose,
    certify_rational_map,
    coeffs_from_map,
    keller_residual_exact,
    keller_target_residual_exact,
    map_from_coeffs,
    n_plane_coeffs,
    propose_and_certify,
    rationalize_coeffs,
)


def test_identity_has_zero_keller_residual_and_target_one() -> None:
    shear = shear_map(0, (0,))  # (x, y)
    # pad to degree 1 via coeffs_from_map
    coeffs = coeffs_from_map(shear, max_degree=1)
    comps = map_from_coeffs(coeffs, max_degree=1)
    assert keller_residual_exact(comps) == 0
    assert keller_target_residual_exact(comps, 1) == 0


def test_certify_shear_is_not_a_violator() -> None:
    shear = shear_map(0, (0, 0, 1))  # (x, y + x^2)
    coeffs = coeffs_from_map(shear, max_degree=2)
    result = certify_rational_map(coeffs, max_degree=2)
    assert result.jacobian_nonzero_constant is True
    assert result.violator is False
    assert result.honesty["jacobian_n2_claim"] is False
    assert result.honesty["jacobian_conjecture_proof_claim"] is False


def test_anneal_smoke_improves_or_stays_near_identity() -> None:
    dim = n_plane_coeffs(1)
    assert dim == 6
    proposal = anneal_keller_propose(max_degree=1, steps=80, seed=7, target=1.0)
    assert len(proposal.coeffs_float) == dim
    assert "keller" in proposal.losses
    # identity start should keep soft keller small
    assert proposal.losses["keller"] < 1.0


def test_propose_and_certify_never_forges_parent() -> None:
    report = propose_and_certify(max_degree=1, steps=60, seed=3, height=2)
    assert report["parent"] == "jacobian_conjecture_n2"
    assert report["certified"]["honesty"]["jacobian_conjecture_proof_claim"] is False
    if not report["certified"]["violator"]:
        assert report["escalate_parent"] is False
        assert report["parent_status"] == "open"


def test_certify_keeps_proof_claim_unearned() -> None:
    from omnibias.holonomic.jacobian_n2 import JACOBIAN_CONJECTURE_PROOF_CLAIM_ALLOWED

    assert JACOBIAN_CONJECTURE_PROOF_CLAIM_ALLOWED is False
    shear = shear_map(0, (0, 0, 1))
    coeffs = coeffs_from_map(shear, max_degree=2)
    result = certify_rational_map(coeffs, max_degree=2)
    assert result.honesty["jacobian_conjecture_proof_claim"] is False
    assert result.payload["honesty"]["jacobian_conjecture_proof_claim"] is False


def test_rationalize_bounds_height() -> None:
    snapped = rationalize_coeffs([0.1, 99.0, -0.01], height=2, denom_cap=10)
    assert all(isinstance(c, Fraction) for c in snapped)
    assert abs(snapped[1]) <= 2 or abs(snapped[1].numerator) <= 2 * snapped[1].denominator
