# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Proof-machine adapter for proof-carrying neural PDE certificates."""

from __future__ import annotations

from typing import Any

from omnibias.core.proof import Certificate, Conjecture, ProofAttempt
from omnibias.core.verified.pde_certificate import (
    BoundaryFace,
    LinearPDE,
    aposteriori_error_certificate,
    pinn_aposteriori_schema_errors,
    replay_pinn_aposteriori_certificate,
)


def _blocked(detail: str) -> ProofAttempt:
    return ProofAttempt(status="BLOCKED", certificate=None, obligations=(detail,), detail=detail)


def _certificate_from_data(data: dict[str, Any]) -> Certificate:
    cert = data.get("certificate")
    if isinstance(cert, dict):
        return cert
    layers = data["layers"]
    domain = data["domain"]
    pde = data["pde"]
    if not isinstance(pde, LinearPDE):
        raise TypeError("data['pde'] must be a LinearPDE when no certificate is supplied")
    boundary = tuple(data.get("boundary", ()))
    if not all(isinstance(face, BoundaryFace) for face in boundary):
        raise TypeError("data['boundary'] must contain BoundaryFace objects")
    built = aposteriori_error_certificate(
        layers,
        domain,
        pde,
        boundary=boundary,
        stability=data.get("stability"),
        stability_interior=float(data.get("stability_interior", 1.0)),
        stability_boundary=float(data.get("stability_boundary", 1.0)),
        invariants=tuple(data.get("invariants", ())),
        max_error=data.get("max_error"),
        splits=data.get("splits", 1),
        boundary_splits=data.get("boundary_splits", 1),
    )
    return built.certificate


def prove_pinn_aposteriori(conjecture: Conjecture) -> ProofAttempt:
    """Adjudicate a sealed a-posteriori neural PDE certificate."""
    try:
        cert = _certificate_from_data(dict(conjecture.data))
    except (KeyError, TypeError, ValueError) as exc:
        return _blocked(f"could not build PINN a-posteriori certificate: {exc}")

    payload = cert.get("payload", {})
    if not isinstance(payload, dict):
        return ProofAttempt(status="BLOCKED", certificate=cert, obligations=("payload is invalid",))
    threshold = conjecture.data.get("max_error")
    if threshold is None and isinstance(payload.get("finite_obligation"), dict):
        threshold = payload["finite_obligation"].get("threshold")

    if threshold is not None and float(payload.get("error_bound", float("inf"))) > float(threshold):
        return ProofAttempt(
            status="BLOCKED",
            certificate=cert,
            obligations=("certified error bound exceeds requested threshold",),
            detail="a-posteriori bound is finite but above threshold",
        )

    return ProofAttempt(
        status="PROVED",
        certificate=cert,
        detail="a-posteriori neural PDE error bound certified for the stated model problem",
    )


def pinn_aposteriori_proof_schema_errors(cert: Certificate) -> list[str]:
    return pinn_aposteriori_schema_errors(cert)


def replay_pinn_aposteriori(cert: Certificate) -> bool:
    return replay_pinn_aposteriori_certificate(cert)


__all__ = [
    "pinn_aposteriori_proof_schema_errors",
    "prove_pinn_aposteriori",
    "replay_pinn_aposteriori",
]
