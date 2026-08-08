# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""omnibias.pinn.domain: non-box geometry via SDFs and R-functions.

Named ``domain`` (not ``geometry``) to avoid colliding with the
``omnibias-geometry`` manifold package.

Backend-free SDF primitives, R-function CSG, normalized ADFs, and
SDF-aware sampling live under ``_core``. The
:class:`~omnibias.pinn.domain.torch.field.DistanceConstrainedField` cage
(``u = g + phi * NN``) lives under ``omnibias.pinn.domain.torch`` /
``.jax``.

Maturity: **alpha** submodule of the Beta ``omnibias-pinn`` distribution.

Honesty
-------
ADF normalization of a general SDF uses a numerical gradient unless an
analytic ``grad_fn`` is supplied. The hard-BC ansatz is exact on the
zero level set of ``phi``; residual accuracy elsewhere is optimised, not
proven.
"""

from __future__ import annotations

from omnibias.pinn.domain._core import (
    Box,
    Cylinder,
    Halfspace,
    Negate,
    Polygon,
    RCompose,
    SDF,
    Sphere,
    approximate_distance,
    boundary_points_sdf,
    complement,
    evaluate_sdf,
    fd_gradient,
    interior_points_sdf,
    intersect,
    normalize_adf,
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
