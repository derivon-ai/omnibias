# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Deterministic proof-carrying PDE smoke benchmark."""

from __future__ import annotations

from typing import Any

from omnibias.core.proof import Conjecture
from omnibias.core.proof.certificate import verify_certificate_digest
from omnibias.core.verified.pde_certificate import (
    aposteriori_error_certificate,
    laplace,
    structural_invariant,
    user_stability_estimate,
)
from omnibias.pinn.certified import build_default_machine


def evaluate_benchmark() -> dict[str, Any]:
    """Certify that an affine harmonic network has zero Laplace residual."""
    layers = [([[2.0, -3.0]], [1.0], None)]
    domain = [(-1.0, 1.0), (-1.0, 1.0)]
    stability = user_stability_estimate(
        0.5,
        1.0,
        source="manufactured harmonic fixture",
        pde_family="laplace",
        domain="unit square",
        assumptions=("linear well-posed model problem",),
    )
    invariant = structural_invariant(
        "harmonic_affine",
        "Delta(2x - 3y + 1) = 0",
        assumptions=("second derivatives of affine functions vanish",),
    )
    cert = aposteriori_error_certificate(
        layers,
        domain,
        laplace(2),
        stability=stability,
        invariants=[invariant],
        max_error=1e-6,
        splits=2,
    )
    machine = build_default_machine()
    verdict = machine.evaluate(
        Conjecture(
            "affine harmonic PDE certificate",
            "pinn_aposteriori_error",
            {"certificate": cert.certificate, "max_error": 1e-6},
        )
    )
    payload = cert.certificate["payload"]
    return {
        "error_bound": cert.error_bound,
        "interior_residual": cert.interior_residual,
        "boundary_residual": cert.boundary_residual,
        "digest_ok": verify_certificate_digest(cert.certificate),
        "verdict": verdict.status,
        "schema_ok": verdict.schema_ok,
        "replay_ok": verdict.replay_ok,
        "honesty_ok": verdict.honesty_ok,
        "unproven_claim": cert.certificate["honesty"]["unproven_claim"],
        "finite_obligation": payload.get("finite_obligation"),
    }


__all__ = ["evaluate_benchmark"]
