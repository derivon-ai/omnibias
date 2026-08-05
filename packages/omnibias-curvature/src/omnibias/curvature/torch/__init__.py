# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Torch, multi-layer exact-curvature sharpness.

The arbitrary-depth, PyTorch counterpart of :mod:`omnibias.curvature.sharpness`
(one-layer, JAX). Curvature functionals are computed matrix-free from exact
Hessian-vector products, so they scale to deep networks and drop into any
training loop -- an MLP, a `JetMLP` PINN, or the `omnibias.score.flow` CNF velocity
field.

Import as::

    from omnibias.curvature.torch import sharpness_aware_loss, sam_objective

This subpackage requires the optional ``torch`` extra
(``pip install "omnibias-curvature[torch]"``); it is intentionally *not*
imported by the JAX-only top-level :mod:`omnibias.curvature`.
"""

from __future__ import annotations

from omnibias.curvature.torch.optim import ExactSAM
from omnibias.curvature.torch.regularize import (
    CollapseResult,
    min_norm_solve,
    numerical_rank,
    rank_collapse,
    regularization_path,
    regularized_solve,
)
from omnibias.curvature.torch.sharpness import (
    HessianOperator,
    curvature_sharpness,
    dense_hessian,
    hessian_eigenvalue_extremes,
    hessian_frobenius_sq,
    hessian_top_eigenvalue,
    hessian_trace,
    hutchinson_frobenius_sq,
    hutchinson_trace,
    hvp,
    sam_objective,
    sam_sharpness_gap,
    sharpness_aware_loss,
    top_eigenvalue,
)

__all__ = [
    "CollapseResult",
    "ExactSAM",
    "HessianOperator",
    "curvature_sharpness",
    "dense_hessian",
    "hessian_eigenvalue_extremes",
    "hessian_frobenius_sq",
    "hessian_top_eigenvalue",
    "hessian_trace",
    "hutchinson_frobenius_sq",
    "hutchinson_trace",
    "hvp",
    "min_norm_solve",
    "numerical_rank",
    "rank_collapse",
    "regularization_path",
    "regularized_solve",
    "sam_objective",
    "sam_sharpness_gap",
    "sharpness_aware_loss",
    "top_eigenvalue",
]
