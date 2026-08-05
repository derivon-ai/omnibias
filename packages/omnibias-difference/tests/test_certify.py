# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""W2: certified derivative -> sealed v1 certificate -> Lean-kernel bridge."""

from __future__ import annotations

import copy

from omnibias.core.proof.certificate import schema_errors_v1, verify_certificate_digest
from omnibias.core.proof.lean_check import lean_check_available
from omnibias.difference import (
    certified_derivative_enclosure,
    check_derivative_certificate,
    derivative_sign_certificate,
)


def test_certificate_is_well_formed_and_sealed() -> None:
    enc = certified_derivative_enclosure("tanh", 0.6, 1)  # tanh' > 0
    cert = derivative_sign_certificate(enc)
    assert schema_errors_v1(cert) == []
    assert verify_certificate_digest(cert)
    assert cert["payload"]["type"] == "interval"
    assert cert["meta"]["name"] == "tanh"
    assert cert["meta"]["sign"] == "positive"


def test_tampering_breaks_the_digest() -> None:
    enc = certified_derivative_enclosure("tanh", 0.6, 1)
    cert = derivative_sign_certificate(enc)
    tampered = copy.deepcopy(cert)
    tampered["claim"] = "a wider, false claim"
    assert not verify_certificate_digest(tampered)
    # The bridge refuses a tampered certificate before emitting any Lean.
    from omnibias.core.proof.lean_check import check_certificate

    result = check_certificate(tampered)
    assert not result.verified
    assert "digest" in result.detail.lower()


def test_sign_definite_derivatives_produce_obligations() -> None:
    pos = check_derivative_certificate(certified_derivative_enclosure("tanh", 0.6, 1))
    assert pos.sign == "positive"
    assert pos.obligation_generated
    assert "enclosed_quantity_pos" in (pos.obligation or "")

    neg = check_derivative_certificate(certified_derivative_enclosure("tanh", 0.6, 2))
    assert neg.sign == "negative"
    assert neg.obligation_generated
    assert "enclosed_quantity_neg" in (neg.obligation or "")


def test_straddling_enclosure_has_no_obligation() -> None:
    # tanh''(0) = 0, so the enclosure straddles zero: a documented obligation gap.
    verdict = check_derivative_certificate(certified_derivative_enclosure("tanh", 0.0, 2))
    assert verdict.sign == "indeterminate"
    assert not verdict.obligation_generated
    assert not verdict.theorem_prover_verified  # no finite obligation to check


def test_theorem_prover_verified_is_never_forged() -> None:
    verdict = check_derivative_certificate(certified_derivative_enclosure("tanh", 0.6, 1))
    # The flag equals the bridge's kernel verdict, exactly -- never asserted True.
    assert verdict.theorem_prover_verified == verdict.lean.verified
    if not lean_check_available():
        assert not verdict.lean.available
        assert not verdict.theorem_prover_verified
    assert verdict.sealed_ok


def test_sign_definite_round_trip_across_many_points() -> None:
    zs = [0.3 + 0.1 * i for i in range(8)]  # K = 8 sign-definite (tanh' > 0) points
    for z in zs:
        verdict = check_derivative_certificate(certified_derivative_enclosure("tanh", z, 1))
        assert verdict.sign == "positive"
        assert verdict.obligation_generated
        assert verdict.sealed_ok
        assert verdict.theorem_prover_verified == verdict.lean.verified
