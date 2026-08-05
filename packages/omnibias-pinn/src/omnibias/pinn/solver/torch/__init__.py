# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Torch backend for omnibias.pinn.solver: steady + evolution drivers and integrators."""

from __future__ import annotations

from omnibias.pinn.solver.torch._solution import FieldSolution, GridSolution
from omnibias.pinn.solver.torch.evolution import (
    SemiDiscrete,
    advection_diffusion_semidiscrete,
    burgers_semidiscrete,
    grid_solution,
    heat_semidiscrete,
    method_of_lines,
    reaction_diffusion_semidiscrete,
    solve_evolution,
)
from omnibias.pinn.solver.torch.fields import build_field, freeze_features
from omnibias.pinn.solver.torch.integrators import (
    burgers_jet_step,
    implicit_linear_step,
    linear_jet_step,
    rk4_step,
)
from omnibias.pinn.solver.torch.inverse import InverseSolution, solve_inverse
from omnibias.pinn.solver.torch.spectral import SpectralGrid1D
from omnibias.pinn.solver.torch.steady import (
    OPTIMIZERS,
    solve_least_squares,
    solve_optimize,
    solve_steady,
)

__all__ = [
    "FieldSolution",
    "GridSolution",
    "InverseSolution",
    "OPTIMIZERS",
    "SemiDiscrete",
    "SpectralGrid1D",
    "advection_diffusion_semidiscrete",
    "build_field",
    "burgers_jet_step",
    "burgers_semidiscrete",
    "freeze_features",
    "grid_solution",
    "heat_semidiscrete",
    "implicit_linear_step",
    "linear_jet_step",
    "method_of_lines",
    "reaction_diffusion_semidiscrete",
    "rk4_step",
    "solve_evolution",
    "solve_inverse",
    "solve_least_squares",
    "solve_optimize",
    "solve_steady",
]
