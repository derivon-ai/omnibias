# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Differentiable control environments on the ``DifferentiableEnvironment`` seam (torch).

Bit-identical-in-spirit twin of :mod:`omnibias.control.jax.envs` (the
environments are numerically identical up to backend RNG/FFT round-off,
not bit-identical, since neither uses the closed-form jet tower). See
that module's docstring for the physics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class DoubleGyrePointMass:
    r"""Controlled tracer in the steady double-gyre flow on ``[0, 2] x [0, 1]``."""

    amplitude: float = 0.1
    dt: float = 0.1
    target: tuple[float, float] = (1.5, 0.75)
    state_cost: float = 1.0
    control_cost: float = 0.1
    terminal_cost_weight: float = 5.0
    state_dim: int = 2
    action_dim: int = 2

    def flow(self, p: Tensor) -> Tensor:
        x, y = p[0], p[1]
        a = self.amplitude
        u = -math.pi * a * torch.sin(math.pi * x) * torch.cos(math.pi * y)
        v = math.pi * a * torch.cos(math.pi * x) * torch.sin(math.pi * y)
        return torch.stack([u, v])

    def step(self, y: Tensor, u: Tensor) -> Tensor:
        def deriv(p: Tensor) -> Tensor:
            return self.flow(p) + u

        h = self.dt
        k1 = deriv(y)
        k2 = deriv(y + 0.5 * h * k1)
        k3 = deriv(y + 0.5 * h * k2)
        k4 = deriv(y + h * k3)
        return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def cost(self, y: Tensor, u: Tensor) -> Tensor:
        target = torch.as_tensor(self.target, dtype=y.dtype, device=y.device)
        err = y - target
        return self.state_cost * torch.sum(err * err) + self.control_cost * torch.sum(u * u)

    def terminal_cost(self, y: Tensor) -> Tensor:
        target = torch.as_tensor(self.target, dtype=y.dtype, device=y.device)
        err = y - target
        return self.terminal_cost_weight * torch.sum(err * err)


@dataclass(frozen=True)
class AdvectionDiffusionGrid:
    r"""A small periodic advection-diffusion grid controlled through a point source.

    Wraps :func:`omnibias.pinn.solver.torch.evolution.advection_diffusion_semidiscrete`
    on a :class:`omnibias.pinn.solver.torch.spectral.SpectralGrid1D` (an
    existing PDE stepper). See :class:`omnibias.control.jax.envs.AdvectionDiffusionGrid`
    for the full description; ``n_grid`` must be even.
    """

    n_grid: int = 8
    length: float = 2.0
    diffusivity: float = 0.02
    velocity: float = 0.3
    dt: float = 0.02
    target_profile_scale: float = 1.0
    state_cost: float = 1.0
    control_cost: float = 0.05
    terminal_cost_weight: float = 2.0

    def __post_init__(self) -> None:
        if self.n_grid % 2 != 0:
            raise ValueError(f"n_grid must be even (spectral FFT grid), got {self.n_grid}")

    @property
    def state_dim(self) -> int:
        return self.n_grid

    @property
    def action_dim(self) -> int:
        return 1

    def _grid(self, dtype: torch.dtype, device: torch.device) -> object:
        from omnibias.pinn.solver.torch.spectral import SpectralGrid1D

        return SpectralGrid1D(self.n_grid, self.length, dtype=dtype, device=device)

    def _target(self, dtype: torch.dtype, device: torch.device) -> Tensor:
        xs = torch.linspace(
            0.0, self.length, self.n_grid + 1, dtype=dtype, device=device
        )[:-1]
        return self.target_profile_scale * torch.sin(math.pi * xs / self.length)

    def step(self, y: Tensor, u: Tensor) -> Tensor:
        from omnibias.pinn.solver.torch.evolution import advection_diffusion_semidiscrete

        forced = y.clone()
        forced[0] = forced[0] + u[0]
        semidiscrete = advection_diffusion_semidiscrete(
            self._grid(y.dtype, y.device), velocity=self.velocity, diffusivity=self.diffusivity
        )
        return forced + self.dt * semidiscrete.rhs(forced)

    def cost(self, y: Tensor, u: Tensor) -> Tensor:
        target = self._target(y.dtype, y.device)
        err = y - target
        return self.state_cost * torch.mean(err * err) + self.control_cost * torch.sum(u * u)

    def terminal_cost(self, y: Tensor) -> Tensor:
        target = self._target(y.dtype, y.device)
        err = y - target
        return self.terminal_cost_weight * torch.mean(err * err)


__all__ = [
    "AdvectionDiffusionGrid",
    "DoubleGyrePointMass",
]
