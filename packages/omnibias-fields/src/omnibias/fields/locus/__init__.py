# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Equality-locus field ops (theory 01-09, gated).

Import the tensor Newton / IFT kernels from
:mod:`omnibias.fields.locus.torch` or :mod:`omnibias.fields.locus.jax`.
The locus is a constraint manifold, not a general closed-form PDE solver.
"""

from __future__ import annotations

from omnibias.core.locus import (
    AffineSet,
    EqualitySystem,
    NewtonResult,
    UnitTerm,
    affine_locus,
    branch_signature,
    certify_locus_point,
    dF_d_weights,
    hessian_blocks,
    is_transversal,
    jacobian,
    locus_tangent,
    newton_project,
    residual,
)

__all__ = [
    "AffineSet",
    "AnsatzSolutionField",
    "EqualityLocusLayer",
    "EqualitySystem",
    "LocusOutput",
    "NewtonResult",
    "UnitTerm",
    "affine_locus",
    "branch_signature",
    "certify_locus_point",
    "dF_d_weights",
    "hessian_blocks",
    "is_transversal",
    "jacobian",
    "locus_tangent",
    "newton_project",
    "residual",
]


def __getattr__(name: str) -> object:
    if name in {"EqualityLocusLayer", "AnsatzSolutionField", "LocusOutput"}:
        from omnibias.fields.locus import torch as _torch

        return getattr(_torch, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name}")
