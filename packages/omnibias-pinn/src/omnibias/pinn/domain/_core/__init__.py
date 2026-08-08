# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Backend-free SDF / ADF / sampling schemas for omnibias.pinn.domain."""

from __future__ import annotations

from omnibias.pinn.domain._core.adf import (
    approximate_distance,
    fd_gradient,
    normalize_adf,
)
from omnibias.pinn.domain._core.sampling import (
    boundary_points_sdf,
    interior_points_sdf,
)
from omnibias.pinn.domain._core.sdf import (
    Box,
    Cylinder,
    Halfspace,
    Negate,
    Polygon,
    RCompose,
    SDF,
    Sphere,
    complement,
    evaluate_sdf,
    intersect,
    r_conjunction,
    r_disjunction,
    r_negation,
    union,
)

__all__ = [
    "Box",
    "Cylinder",
    "Halfspace",
    "Negate",
    "Polygon",
    "RCompose",
    "SDF",
    "Sphere",
    "approximate_distance",
    "boundary_points_sdf",
    "complement",
    "evaluate_sdf",
    "fd_gradient",
    "interior_points_sdf",
    "intersect",
    "normalize_adf",
    "r_conjunction",
    "r_disjunction",
    "r_negation",
    "union",
]
