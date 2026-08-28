# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Differentiable control environments on the ``DifferentiableEnvironment`` seam (JAX).

Two environments, both built so any trainer written against
:class:`omnibias.control.ocp.DifferentiableEnvironment` runs unchanged:

* :class:`DoubleGyrePointMass` -- a controlled tracer in the steady
  double-gyre velocity field (Shadden et al.), state ``(x, y)``, action a
  velocity correction. Cheap (``state_dim = 2``), used for the adjoint
  exactness and policy-gradient gates.
* :class:`AdvectionDiffusionGrid` -- a genuinely PDE-grid environment: a
  concentration field advected by the same double-gyre velocity and
  diffused, stepped by
  :func:`omnibias.pinn.solver.jax.evolution.advection_diffusion_semidiscrete`
  (an *existing* PDE stepper, not a new one), with a boundary-source
  control input. Kept to a small grid deliberately -- the "cost at PDE
  scale" risk in ``theory/10-control/02-jet-adjoint-policy-optimization.md``
  is the ``O(state_dim)`` cost of one directional jet per state dimension
  in :func:`omnibias.control.jax.adjoint.policy_jacobian_dy`, and this
  environment is the one used to *measure* that cost, not to hide it.

Both environments accept an un-batched state (shape ``(state_dim,)``); use
``jax.vmap`` at the call site for batched rollouts (as
:mod:`omnibias.control.jax.rollout` already does for the CBF filter).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array


@dataclass(frozen=True)
class DoubleGyrePointMass:
    r"""Controlled tracer in the steady double-gyre flow on ``[0, 2] x [0, 1]``.

    .. math::
        u(x, y) = -\pi A \sin(\pi x)\cos(\pi y), \qquad
        v(x, y) = \pi A \cos(\pi x)\sin(\pi y)

    ``step`` integrates ``dot{p} = flow(p) + action`` for one ``dt`` with a
    single RK4 stage (the flow is smooth, so RK4 is exact to machine
    precision at these step sizes for the gate tolerances used here).
    Reward is quadratic tracking of ``target`` plus a control-effort term.
    """

    amplitude: float = 0.1
    dt: float = 0.1
    target: tuple[float, float] = (1.5, 0.75)
    state_cost: float = 1.0
    control_cost: float = 0.1
    terminal_cost_weight: float = 5.0
    state_dim: int = 2
    action_dim: int = 2

    def flow(self, p: Array) -> Array:
        x, y = p[0], p[1]
        a = self.amplitude
        u = -jnp.pi * a * jnp.sin(jnp.pi * x) * jnp.cos(jnp.pi * y)
        v = jnp.pi * a * jnp.cos(jnp.pi * x) * jnp.sin(jnp.pi * y)
        return jnp.stack([u, v])

    def step(self, y: Array, u: Array) -> Array:
        def deriv(p: Array) -> Array:
            return self.flow(p) + u

        h = self.dt
        k1 = deriv(y)
        k2 = deriv(y + 0.5 * h * k1)
        k3 = deriv(y + 0.5 * h * k2)
        k4 = deriv(y + h * k3)
        return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def cost(self, y: Array, u: Array) -> Array:
        target = jnp.asarray(self.target, dtype=y.dtype)
        err = y - target
        return self.state_cost * jnp.sum(err * err) + self.control_cost * jnp.sum(u * u)

    def terminal_cost(self, y: Array) -> Array:
        target = jnp.asarray(self.target, dtype=y.dtype)
        err = y - target
        return self.terminal_cost_weight * jnp.sum(err * err)


@dataclass(frozen=True)
class AdvectionDiffusionGrid:
    r"""A small periodic advection-diffusion grid controlled through a point source.

    Wraps :func:`omnibias.pinn.solver.jax.evolution.advection_diffusion_semidiscrete`
    on a :class:`omnibias.pinn.solver.jax.spectral.SpectralGrid1D` (an
    *existing* PDE stepper, not a new one) with a constant advection
    velocity and diffusivity. The control action is a scalar source added
    to grid point ``0`` before each explicit-Euler step.

    ``state_dim = n_grid`` deliberately stays small (default 8, and must
    be even -- the spectral grid's FFT convention): the adjoint trainer's
    ``dpi/dy`` assembly costs one directional jet per state dimension, and
    this environment is where that cost is *measured*
    (``benchmarks/adjoint_exactness.py``), not hidden.
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

    def _grid(self) -> object:
        from omnibias.pinn.solver.jax.spectral import SpectralGrid1D

        return SpectralGrid1D(self.n_grid, self.length)

    def _target(self, dtype: object) -> Array:
        xs = jnp.linspace(0.0, self.length, self.n_grid, endpoint=False, dtype=dtype)
        return self.target_profile_scale * jnp.sin(jnp.pi * xs / self.length)

    def step(self, y: Array, u: Array) -> Array:
        from omnibias.pinn.solver.jax.evolution import advection_diffusion_semidiscrete

        forced = y.at[0].add(u[0])
        semidiscrete = advection_diffusion_semidiscrete(
            self._grid(), velocity=self.velocity, diffusivity=self.diffusivity
        )
        return forced + self.dt * semidiscrete.rhs(forced)

    def cost(self, y: Array, u: Array) -> Array:
        target = self._target(y.dtype)
        err = y - target
        return self.state_cost * jnp.mean(err * err) + self.control_cost * jnp.sum(u * u)

    def terminal_cost(self, y: Array) -> Array:
        target = self._target(y.dtype)
        err = y - target
        return self.terminal_cost_weight * jnp.mean(err * err)


__all__ = [
    "AdvectionDiffusionGrid",
    "DoubleGyrePointMass",
]
