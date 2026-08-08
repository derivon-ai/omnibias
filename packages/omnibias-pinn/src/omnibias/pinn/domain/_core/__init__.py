# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Backend-free SDF / ADF / sampling schemas for omnibias.pinn.domain."""

from __future__ import annotations

from omnibias.pinn.domain._core.adf import (
    approximate_distance,
    fd_gradient,
    normalize_adf,
)
from omnibias.pinn.domain._core.boundary import (
    BCMode,
    NonSmoothBoundaryError,
    assert_smooth_for_normal_bc,
    bc_distance_factor,
    boundary_factor_jet,
    boundary_junction_mask,
    normalized_boundary_factor,
    omega_gradient,
)
from omnibias.pinn.domain._core.sampling import (
    boundary_points_sdf,
    interior_points_sdf,
)
from omnibias.pinn.domain._core.sdf import (
    SDF,
    Box,
    Cylinder,
    Halfspace,
    Negate,
    Polygon,
    RCompose,
    Sphere,
    complement,
    evaluate_sdf,
    intersect,
    r_conjunction,
    r_disjunction,
    r_intersect_sdf,
    r_negation,
    r_union_sdf,
    union,
)

__all__ = [
    "BCMode",
    "Box",
    "Cylinder",
    "Halfspace",
    "Negate",
    "NonSmoothBoundaryError",
    "Polygon",
    "RCompose",
    "SDF",
    "Sphere",
    "approximate_distance",
    "assert_smooth_for_normal_bc",
    "bc_distance_factor",
    "boundary_factor_jet",
    "boundary_junction_mask",
    "boundary_points_sdf",
    "complement",
    "evaluate_sdf",
    "fd_gradient",
    "interior_points_sdf",
    "intersect",
    "normalize_adf",
    "normalized_boundary_factor",
    "omega_gradient",
    "r_conjunction",
    "r_disjunction",
    "r_intersect_sdf",
    "r_negation",
    "r_union_sdf",
    "union",
]
