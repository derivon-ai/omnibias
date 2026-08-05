# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Functional operator surface for the torch backend (Option 2 kernel).

The :class:`ComponentView` / :class:`VectorView` attribute DSL forwards
into these functions; users may also import them directly when writing
programmatic / extension code:

    from omnibias.pinn.torch import ops
    adv = ops.advection(state, velocity=("u", "v", "w"))

All ops consume a :class:`FieldState` (never raw ``(u, coords)``
tensors) so the closed-form path stays closed-form: the state's
:class:`SigmaCache` is filled lazily and reused across orders.
"""

from __future__ import annotations

from omnibias.pinn.torch.ops.basic import (
    derivative,
    divergence,
    gradient,
    laplacian,
    mixed_partial,
    stack_components,
    value,
    vector_derivative,
)
from omnibias.pinn.torch.ops.high_order import (
    biharmonic,
    gradient_of_derivative,
    hessian,
    jacobian,
    polylaplacian,
    spatial_hessian,
    vector_biharmonic,
    vector_hessian,
    vector_laplacian,
    vector_polylaplacian,
)
from omnibias.pinn.torch.ops.nonlinear import (
    advection,
    directional_derivative,
    material_derivative,
    p_laplacian,
)
from omnibias.pinn.torch.ops.registry import register
from omnibias.pinn.torch.ops.vector import (
    curl,
    deformation_gradient,
    spatial_jacobian,
    strain_rate,
    vorticity,
)

__all__ = [
    "advection",
    "biharmonic",
    "curl",
    "deformation_gradient",
    "derivative",
    "directional_derivative",
    "divergence",
    "gradient",
    "gradient_of_derivative",
    "hessian",
    "jacobian",
    "laplacian",
    "material_derivative",
    "mixed_partial",
    "p_laplacian",
    "polylaplacian",
    "register",
    "spatial_hessian",
    "spatial_jacobian",
    "stack_components",
    "strain_rate",
    "value",
    "vector_biharmonic",
    "vector_derivative",
    "vector_hessian",
    "vector_laplacian",
    "vector_polylaplacian",
    "vorticity",
]
