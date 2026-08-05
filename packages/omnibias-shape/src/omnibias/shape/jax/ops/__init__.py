# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""JAX soft-shape / coverage operator surface (twin of ``omnibias.shape.torch.ops``)."""

from __future__ import annotations

from omnibias.shape.jax.ops.cardinality import anneal_lambda, l0_surrogate, prune_inactive
from omnibias.shape.jax.ops.coverage import (
    CoverageCache,
    coverage_energy,
    coverage_energy_grad,
    coverage_energy_hessian,
    coverage_residual,
    lse_coverage,
    soft_or_coverage,
)
from omnibias.shape.jax.ops.occupancy import (
    soft_box,
    soft_box_grad,
    soft_box_hessian,
    soft_disk,
    soft_interval,
    soft_polytope,
)

__all__ = [
    "CoverageCache",
    "anneal_lambda",
    "coverage_energy",
    "coverage_energy_grad",
    "coverage_energy_hessian",
    "coverage_residual",
    "l0_surrogate",
    "lse_coverage",
    "prune_inactive",
    "soft_box",
    "soft_box_grad",
    "soft_box_hessian",
    "soft_disk",
    "soft_interval",
    "soft_or_coverage",
    "soft_polytope",
]
