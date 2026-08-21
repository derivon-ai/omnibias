# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Finite rational obligation classes for the Lean-kernel bridge.

New payload kinds live here so ``omnibias.core.proof.lean_check`` stays a
driver, not a catalogue. Infinite analytic statements are out of scope
and are not expressed in Lean.
"""

from __future__ import annotations

from omnibias.core.proof.obligations.rational_stencil import (
    PAYLOAD_POISEDNESS,
    PAYLOAD_STENCIL,
    Obligation,
    RationalStencil,
    RationalSupport,
    StencilCertificateReport,
    curated_rational_stencils,
    honesty_payload,
    poisedness_obligation,
    seal_poisedness_certificate,
    seal_stencil_certificate,
    stencil_consistency_obligation,
)

__all__ = [
    "Obligation",
    "PAYLOAD_POISEDNESS",
    "PAYLOAD_STENCIL",
    "RationalStencil",
    "RationalSupport",
    "StencilCertificateReport",
    "curated_rational_stencils",
    "honesty_payload",
    "poisedness_obligation",
    "seal_poisedness_certificate",
    "seal_stencil_certificate",
    "stencil_consistency_obligation",
]
