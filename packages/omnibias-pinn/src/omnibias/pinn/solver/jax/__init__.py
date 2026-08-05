# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""JAX backend for omnibias.pinn.solver: steady + evolution drivers and integrators.

Importing this module enables JAX double precision (``jax_enable_x64``): the
closed-form derivative tower is bit-stable and the torch<->jax parity tests
require float64, matching the rest of the omnibias jax stack.
"""

from __future__ import annotations

import jax as _jax

_jax.config.update("jax_enable_x64", True)

from omnibias.pinn.solver.jax._solution import FieldSolution, GridSolution  # noqa: E402
from omnibias.pinn.solver.jax.evolution import (  # noqa: E402
    SemiDiscrete,
    advection_diffusion_semidiscrete,
    burgers_semidiscrete,
    grid_solution,
    heat_semidiscrete,
    kuramoto_sivashinsky_semidiscrete,
    method_of_lines,
    reaction_diffusion_semidiscrete,
    solve_evolution,
)
from omnibias.pinn.solver.jax.fields import (  # noqa: E402
    build_field,
    field_from_arrays,
    with_readout,
)
from omnibias.pinn.solver.jax.integrators import (  # noqa: E402
    burgers_jet_step,
    implicit_linear_step,
    linear_jet_step,
    rk4_step,
)
from omnibias.pinn.solver.jax.spectral import SpectralGrid1D  # noqa: E402
from omnibias.pinn.solver.jax.steady import solve_least_squares, solve_steady  # noqa: E402
from omnibias.pinn.solver.jax.stiff import (  # noqa: E402
    closed_form_jacobian,
    dense_jacobian,
    etdrk4_step,
    exponential_rosenbrock_step,
    imex_cnab2_step,
    imex_euler_step,
    phi_diagonal,
    phi_matrix,
    rosenbrock_step,
    stiff_rollout,
)

__all__ = [
    "FieldSolution",
    "GridSolution",
    "SemiDiscrete",
    "SpectralGrid1D",
    "advection_diffusion_semidiscrete",
    "build_field",
    "burgers_jet_step",
    "burgers_semidiscrete",
    "closed_form_jacobian",
    "dense_jacobian",
    "etdrk4_step",
    "exponential_rosenbrock_step",
    "field_from_arrays",
    "grid_solution",
    "heat_semidiscrete",
    "imex_cnab2_step",
    "imex_euler_step",
    "implicit_linear_step",
    "kuramoto_sivashinsky_semidiscrete",
    "linear_jet_step",
    "method_of_lines",
    "phi_diagonal",
    "phi_matrix",
    "reaction_diffusion_semidiscrete",
    "rk4_step",
    "rosenbrock_step",
    "solve_evolution",
    "solve_least_squares",
    "solve_steady",
    "stiff_rollout",
    "with_readout",
]
