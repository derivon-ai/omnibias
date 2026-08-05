# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""JAX differential-geometry operator surface."""

from __future__ import annotations

from omnibias.geometry.jax.ops.connection import (
    christoffel,
    einstein_equation_residual,
    einstein_tensor,
    geodesic_deviation,
    geodesic_rhs,
    inverse_metric,
    kretschmann_scalar,
    lowered_riemann,
    metric,
    metric_density_divergence,
    ricci_tensor,
    riemann_tensor,
    scalar_curvature,
    sqrt_det_metric,
    weyl_tensor,
)
from omnibias.geometry.jax.ops.exterior import (
    codifferential,
    codifferential_exact_scalar,
    d_squared_scalar,
    exterior_derivative,
    hodge_laplacian_scalar,
    hodge_star,
    interior_product,
    lie_derivative,
    wedge,
)
from omnibias.geometry.jax.ops.field_ops import (
    covariant_derivative,
    laplace_beltrami,
)
from omnibias.geometry.jax.ops.fisher import (
    exponential_family_fisher_manifold,
    exponential_family_fisher_metric,
)
from omnibias.geometry.jax.ops.integration import (
    integrate_form,
    integrate_form_values,
    surface_area,
    surface_integral,
    volume_element,
)
from omnibias.geometry.jax.ops.pullback import (
    euclidean_ambient_metric,
    metric_spec_from_chart,
    pullback_metric,
)
from omnibias.geometry.jax.ops.topology import (
    betti_number,
    gauss_bonnet_euler,
    harmonic_projection,
    hodge_laplacian,
    hodge_laplacian_matrix,
    map_degree,
    winding_number,
)

__all__ = [
    "betti_number",
    "christoffel",
    "codifferential",
    "codifferential_exact_scalar",
    "covariant_derivative",
    "d_squared_scalar",
    "einstein_equation_residual",
    "einstein_tensor",
    "euclidean_ambient_metric",
    "exponential_family_fisher_manifold",
    "exponential_family_fisher_metric",
    "exterior_derivative",
    "gauss_bonnet_euler",
    "geodesic_deviation",
    "geodesic_rhs",
    "harmonic_projection",
    "hodge_laplacian",
    "hodge_laplacian_matrix",
    "hodge_laplacian_scalar",
    "hodge_star",
    "integrate_form",
    "integrate_form_values",
    "interior_product",
    "inverse_metric",
    "kretschmann_scalar",
    "laplace_beltrami",
    "lie_derivative",
    "lowered_riemann",
    "map_degree",
    "metric",
    "metric_density_divergence",
    "metric_spec_from_chart",
    "pullback_metric",
    "ricci_tensor",
    "riemann_tensor",
    "scalar_curvature",
    "sqrt_det_metric",
    "surface_area",
    "surface_integral",
    "volume_element",
    "wedge",
    "weyl_tensor",
    "winding_number",
]
