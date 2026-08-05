# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""JAX variational / least-action operator surface."""

from __future__ import annotations

from omnibias.variational.jax.ops.action import (
    action,
    integrate_values,
    lagrangian_values,
)
from omnibias.variational.jax.ops.constraints import (
    augmented_lagrangian,
    constrained_euler_lagrange_residual,
)
from omnibias.variational.jax.ops.dynamics import (
    acceleration,
    dynamics_rhs,
    generalized_force,
    inverse_dynamics,
    mass_matrix,
    predicted_acceleration,
)
from omnibias.variational.jax.ops.euler_lagrange import (
    euler_lagrange_residual,
    lagrangian_partials,
    trajectory,
)
from omnibias.variational.jax.ops.field_theory import (
    action_density,
    density_values,
    field_euler_lagrange_residual,
    field_functional_derivative,
    first_variation_density,
    stress_energy_tensor,
)
from omnibias.variational.jax.ops.functional import (
    first_variation,
    functional_derivative,
)
from omnibias.variational.jax.ops.geodesic import geodesic_action, metric_lagrangian
from omnibias.variational.jax.ops.hamiltonian import (
    conjugate_momentum,
    energy,
    hamiltonian,
    hamiltons_equations_residual,
)
from omnibias.variational.jax.ops.integrators import (
    discrete_euler_lagrange_residual,
    stormer_verlet_step,
)
from omnibias.variational.jax.ops.legendre import (
    canonical_equations,
    hamiltonian_from_lagrangian,
    legendre_transform,
    momentum,
    velocity_from_momentum,
)
from omnibias.variational.jax.ops.losses import (
    action_minimization_loss,
    euler_lagrange_loss,
    field_euler_lagrange_loss,
    lagrangian_dynamics_loss,
)
from omnibias.variational.jax.ops.noether import noether_charge

__all__ = [
    "acceleration",
    "action",
    "action_density",
    "action_minimization_loss",
    "augmented_lagrangian",
    "canonical_equations",
    "conjugate_momentum",
    "constrained_euler_lagrange_residual",
    "density_values",
    "discrete_euler_lagrange_residual",
    "dynamics_rhs",
    "energy",
    "euler_lagrange_loss",
    "euler_lagrange_residual",
    "field_euler_lagrange_loss",
    "field_euler_lagrange_residual",
    "field_functional_derivative",
    "first_variation",
    "first_variation_density",
    "functional_derivative",
    "generalized_force",
    "geodesic_action",
    "hamiltonian",
    "hamiltonian_from_lagrangian",
    "hamiltons_equations_residual",
    "integrate_values",
    "inverse_dynamics",
    "lagrangian_dynamics_loss",
    "lagrangian_partials",
    "lagrangian_values",
    "legendre_transform",
    "mass_matrix",
    "metric_lagrangian",
    "momentum",
    "noether_charge",
    "predicted_acceleration",
    "stormer_verlet_step",
    "stress_energy_tensor",
    "trajectory",
    "velocity_from_momentum",
]
