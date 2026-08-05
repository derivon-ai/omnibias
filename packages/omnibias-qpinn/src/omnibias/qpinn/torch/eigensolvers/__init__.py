# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Closed-form eigenvalue solvers built on omnibias derivatives.

The :mod:`galerkin` submodule provides a *direct* generalized
eigenvalue solver that bypasses Adam for the eigenvalue itself: a
trainable :math:`K`-channel basis network feeds bit-stable
closed-form Laplacian + potential values into Galerkin matrix
construction, then :func:`scipy.linalg.eigh` (or
:func:`torch.linalg.eigh` after Cholesky symmetrisation) extracts the
:math:`K` lowest eigenpairs in one call. This is the SOTA path for
bound-state eigenvalue problems (NH3 inversion tunneling, double-well
QHO, etc.) where the v0.0.2a1 Rayleigh-quotient + Adam pipeline failed
to converge to spectroscopic precision.

Backend: **torch only** (alpha). The Galerkin assembly is tied to the torch
field basis + closed-form Laplacian and :func:`scipy.linalg.eigh`; a JAX twin is
on the roadmap, so ``omnibias.qpinn.jax`` intentionally ships no ``eigensolvers``
submodule (rather than a silent partial one). The equation / cage / diagnostics
residuals are bit-identical across both backends.
"""

from __future__ import annotations

from omnibias.qpinn.torch.eigensolvers.galerkin import (
    GalerkinEigenResult,
    galerkin_eigh,
    galerkin_eigh_real_basis,
    galerkin_matrices,
    galerkin_trace_loss,
)

__all__ = [
    "GalerkinEigenResult",
    "galerkin_eigh",
    "galerkin_eigh_real_basis",
    "galerkin_matrices",
    "galerkin_trace_loss",
]
