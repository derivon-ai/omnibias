# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Certified selection (measure-mode collapse): the backend-agnostic public surface.

Re-exports the numpy certificate surface of :mod:`omnibias.struct._core.select` at the struct
root (mirroring how :func:`omnibias.struct.certify_soft_dp` sits alongside the DP layers). The
differentiable soft-selection ops and Gibbs moments live in the backend twins
:mod:`omnibias.struct.torch.select` / :mod:`omnibias.struct.jax.select`.

This is the ``beta -> inf`` *feasibility / temperature* collapse of a Gibbs law onto its mode --
**not** the founding ``delta -> 0`` bias collapse; the derivative tower is only the exact engine
that differentiates ``lse_beta`` (do not conflate the two axes).
"""

from __future__ import annotations

from omnibias.struct._core.select import (
    SelectionCertificate,
    argmax_stability_margin,
    beta_for_confidence,
    certify_argmax,
    mass_concentration_bound,
    seal_selection_certificate,
)

__all__ = [
    "SelectionCertificate",
    "argmax_stability_margin",
    "beta_for_confidence",
    "certify_argmax",
    "mass_concentration_bound",
    "seal_selection_certificate",
]
