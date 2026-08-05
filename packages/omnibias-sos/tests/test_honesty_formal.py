# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Sealing, honesty labels, tamper-evidence, and graceful Lean degradation."""

from __future__ import annotations

import pytest
from omnibias.core.proof.certificate import schema_errors_v1, verify_certificate_digest
from omnibias.sos.certify import certify_sos
from omnibias.sos.formal import (
    drive_sos_obligation,
    is_theorem_prover_verified,
    lean_available,
    lean_check_sos,
)
from omnibias.sos.honesty import (
    FINITE_DIM_SYSTEM,
    GALERKIN_TRUNCATION,
    GLOBAL_POLYNOMIAL,
    SOSScope,
    honesty_labels,
    seal_sos_certificate,
)
from omnibias.sos.problem import Polynomial

X = Polynomial.variable(0, 2)
Y = Polynomial.variable(1, 2)
ONE = Polynomial.constant(1.0, 2)
PD_POLY = X * X - X * Y + Y * Y + ONE


def _sealed():
    cert = certify_sos(PD_POLY)
    assert cert.certified
    return cert, seal_sos_certificate(cert, claim="p(x,y) = x^2 - xy + y^2 + 1 >= 0 for all (x,y)")


def test_sealed_certificate_is_valid_and_makes_no_unproven_claim() -> None:
    _cert, sealed = _sealed()
    assert schema_errors_v1(sealed) == []
    assert verify_certificate_digest(sealed)
    assert sealed["honesty"]["unproven_claim"] is False
    assert sealed["honesty"]["continuum_pde_claim"] is False
    assert sealed["honesty"]["regularity_claim"] is False
    assert sealed["payload"]["type"] == "positive_definite"
    assert sealed["payload"]["pivots"]
    assert sealed["meta"]["generator"] == "omnibias-sos"


@pytest.mark.parametrize(
    ("scope", "finite"),
    [
        (SOSScope(GLOBAL_POLYNOMIAL), False),
        (SOSScope(FINITE_DIM_SYSTEM, system="van der Pol"), True),
        (SOSScope(GALERKIN_TRUNCATION, truncation_order=4, system="2-D fluid"), True),
    ],
)
def test_scope_labels_are_honest(scope: SOSScope, finite: bool) -> None:
    labels = honesty_labels(scope)
    assert labels["unproven_claim"] is False
    assert labels["finite_dimensional"] is finite
    cert = certify_sos(PD_POLY)
    sealed = seal_sos_certificate(cert, claim="bound", scope=scope)
    assert sealed["honesty"]["unproven_claim"] is False
    assert sealed["meta"]["sos"]["scope"] == scope.kind
    assert sealed["meta"]["sos"]["truncation_order"] == scope.truncation_order


def test_no_scope_can_emit_unproven_claim_true() -> None:
    # Every valid scope, and the default, must produce unproven_claim == False.
    for kind in (GLOBAL_POLYNOMIAL, FINITE_DIM_SYSTEM, GALERKIN_TRUNCATION):
        assert honesty_labels(SOSScope(kind))["unproven_claim"] is False


def test_invalid_scope_rejected() -> None:
    with pytest.raises(ValueError, match="scope kind"):
        SOSScope("navier_stokes_regularity")


def test_cannot_seal_inconclusive() -> None:
    motzkin = Polynomial(2, {(4, 2): 1.0, (2, 4): 1.0, (2, 2): -3.0, (0, 0): 1.0})
    cert = certify_sos(motzkin)
    assert not cert.certified
    with pytest.raises(ValueError, match="inconclusive"):
        seal_sos_certificate(cert, claim="should not seal")


def test_tamper_evidence_blocks_lean() -> None:
    _cert, sealed = _sealed()
    tampered = dict(sealed)
    tampered["payload"] = dict(sealed["payload"])
    pivots = [dict(p) for p in sealed["payload"]["pivots"]]
    pivots[0]["lo"] = (0.5).hex()  # forge a "better" margin
    tampered["payload"]["pivots"] = pivots
    assert not verify_certificate_digest(tampered)
    result = lean_check_sos(tampered)
    assert result.verified is False
    assert "digest" in result.detail.lower()


def test_lean_check_degrades_gracefully() -> None:
    _cert, sealed = _sealed()
    result = lean_check_sos(sealed)
    if not lean_available():
        assert result.available is False
        assert result.verified is False
        assert not is_theorem_prover_verified(sealed)
    else:  # pragma: no cover - only when a Lean toolchain is installed
        assert result.available is True


def test_drive_sos_obligation_is_optional() -> None:
    _cert, sealed = _sealed()
    report = drive_sos_obligation(sealed)
    # None when omnibias-formal is not installed; otherwise a DriveReport that
    # never conflates its Mathlib tier with theorem_prover_verified or an unproven-result claim.
    if report is not None:  # pragma: no cover - only when omnibias-formal present
        assert report.tier in (None, "mathlib_verified")
