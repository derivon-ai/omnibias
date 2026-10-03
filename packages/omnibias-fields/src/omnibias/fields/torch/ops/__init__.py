# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Functional operator surface for the torch backend (Option 2 kernel).

The :class:`ComponentView` / :class:`VectorView` attribute DSL forwards
into these functions; users may also import them directly when writing
programmatic / extension code:

    from omnibias.fields.torch import ops
    adv = ops.advection(state, velocity=("u", "v", "w"))

All ops consume a :class:`FieldState` (never raw ``(u, coords)``
tensors) so the closed-form path stays closed-form: the state's
:class:`SigmaCache` is filled lazily and reused across orders.
"""

from __future__ import annotations

from omnibias.fields.torch.ops.basic import (
    derivative,
    divergence,
    gradient,
    laplacian,
    mixed_partial,
    stack_components,
    value,
    vector_derivative,
)
from omnibias.fields.torch.ops.complex import dz, dzbar
from omnibias.fields.torch.ops.conservation import (
    advection_diffusion_residual,
    conservation_residual,
    dalembertian,
    diffusive_flux,
    flux_divergence,
    grad_squared_norm,
    gradient_of_composition,
    laplacian_of_composition,
    variable_coefficient_diffusion,
    wave_operator,
)
from omnibias.fields.torch.ops.high_order import (
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
from omnibias.fields.torch.ops.integral import (
    integrate,
    line_integral,
    quadrature_nodes,
)
from omnibias.fields.torch.ops.nonlinear import (
    advection,
    directional_derivative,
    material_derivative,
    p_laplacian,
    skew_symmetric_advection,
)
from omnibias.fields.torch.ops.norms import inner_product, l2_norm, sobolev_norm
from omnibias.fields.torch.ops.registry import register
from omnibias.fields.torch.ops.tensor import (
    tensor_cofactor,
    tensor_determinant,
    tensor_divergence,
    tensor_double_dot,
    tensor_inverse,
    tensor_matmul,
    tensor_trace,
    tensor_transpose,
)
from omnibias.fields.torch.ops.vector import (
    curl,
    curl_of_curl,
    deformation_gradient,
    gradient_of_divergence,
    rate_of_rotation_tensor,
    spatial_jacobian,
    strain_rate,
    vorticity,
)

__all__ = [
    "advection",
    "advection_diffusion_residual",
    "biharmonic",
    "conservation_residual",
    "curl",
    "curl_of_curl",
    "dalembertian",
    "deformation_gradient",
    "derivative",
    "diffusive_flux",
    "directional_derivative",
    "divergence",
    "dz",
    "dzbar",
    "flux_divergence",
    "grad_squared_norm",
    "gradient",
    "gradient_of_composition",
    "gradient_of_derivative",
    "gradient_of_divergence",
    "hessian",
    "inner_product",
    "integrate",
    "jacobian",
    "l2_norm",
    "laplacian",
    "laplacian_of_composition",
    "line_integral",
    "material_derivative",
    "mixed_partial",
    "p_laplacian",
    "polylaplacian",
    "quadrature_nodes",
    "rate_of_rotation_tensor",
    "register",
    "skew_symmetric_advection",
    "sobolev_norm",
    "spatial_hessian",
    "spatial_jacobian",
    "stack_components",
    "strain_rate",
    "tensor_cofactor",
    "tensor_determinant",
    "tensor_divergence",
    "tensor_double_dot",
    "tensor_inverse",
    "tensor_matmul",
    "tensor_trace",
    "tensor_transpose",
    "value",
    "variable_coefficient_diffusion",
    "vector_biharmonic",
    "vector_derivative",
    "vector_hessian",
    "vector_laplacian",
    "vector_polylaplacian",
    "vorticity",
]
